#!/usr/bin/env python3
"""
s3_reconstruct_data.py — Stage 3 统一重构脚本 (vLLM)
================================================

正式版约定:
  - Stage 3a (RA) 与 Stage 3b (Routing) 共用同一套重构骨架
  - 差异仅通过 task spec 控制: system prompt / miss-policy / 输出校验
  - 并行切 shard、vLLM 推理、keep/blank 判定、对齐校验、汇总逻辑完全一致
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import BeamSearchParams, SamplingParams

from config.config import stage3_reconstruct_dir

SID_RE = re.compile(r"(<\|sid_begin\|>(?:<s_[a-d]_\d+>){4}<\|sid_end\|>)")


@dataclass(frozen=True)
class TaskSpec:
    task: str
    system_message: str
    required_input_columns: tuple[str, ...]
    required_output_columns: tuple[str, ...]
    blank_think: Optional[str] = None

    def apply_miss_policy(self, row_dict: dict) -> dict:
        modified = dict(row_dict)
        if self.task == "ra":
            modified["title"] = None
            modified["categories"] = None
        elif self.task == "routing":
            modified["sid_routing_think"] = self.blank_think
            modified["cot_steps"] = 0
        else:
            raise ValueError(f"未知 task: {self.task}")
        modified["recon_kept"] = False
        return modified


TASK_SPECS = {
    "ra": TaskSpec(
        task="ra",
        system_message=(
            "You are a professional recommendation expert who needs to recommend "
            "the next possible purchase for users based on their purchase history. "
            "Please predict the most likely next product that the user will purchase "
            "based on the user's historical purchase information."
        ),
        required_input_columns=("description", "groundtruth", "title", "categories"),
        required_output_columns=("description", "groundtruth", "title", "categories", "recon_kept"),
    ),
    "routing": TaskSpec(
        task="routing",
        system_message=(
            "You are a cognitive recommender that navigates through a semantic index hierarchy. "
            "Based on the user's purchase history, predict the next item's Semantic ID. "
            "For simple cases, respond with an empty think block; for harder cases, reason through the SID layers."
        ),
        required_input_columns=("description", "groundtruth", "sid_routing_think", "cot_steps"),
        required_output_columns=("description", "groundtruth", "sid_routing_think", "cot_steps", "recon_kept"),
        blank_think="<think>\n\n</think>",
    ),
}


def extract_sid(text: str) -> Optional[str]:
    m = SID_RE.search(text)
    return m.group(1) if m else None


def normalize_task(task: str | None, config_name: str | None) -> str:
    if task:
        task = task.strip().lower()
        if task in TASK_SPECS:
            return task
    if config_name:
        if config_name.endswith("_ra"):
            return "ra"
        if config_name.endswith("_routing"):
            return "routing"
    raise ValueError("无法确定重构任务类型；请传 --task {ra,routing} 或使用 *_ra / *_routing config_name")


def get_task_spec(task: str) -> TaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as exc:
        raise ValueError(f"未知重构任务: {task}") from exc


def build_prompt(spec: TaskSpec, description: str) -> str:
    return (
        f"<|im_start|>system\n{spec.system_message}<|im_end|>\n"
        f"<|im_start|>user\n{description}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def validate_input_frame(df: pd.DataFrame, spec: TaskSpec) -> None:
    missing = [c for c in spec.required_input_columns if c not in df.columns]
    if missing:
        raise SystemExit(f"输入数据缺少必要列: {missing}")
    if len(df) == 0:
        raise SystemExit("输入数据为空，无法重构")


def validate_output_alignment(input_df: pd.DataFrame, output_df: pd.DataFrame, spec: TaskSpec) -> None:
    if len(input_df) != len(output_df):
        raise SystemExit(f"重构后行数不一致: input={len(input_df)} output={len(output_df)}")

    required = [c for c in spec.required_output_columns if c not in output_df.columns]
    if required:
        raise SystemExit(f"输出数据缺少必要列: {required}")

    for key in ("sample_id", "description", "groundtruth"):
        if key in input_df.columns:
            left = input_df[key].astype(str).tolist()
            right = output_df[key].astype(str).tolist()
            if left != right:
                raise SystemExit(f"重构对齐校验失败: 列 {key} 发生变化")


def reconstruct_records(
    *,
    input_df: pd.DataFrame,
    sid_outputs,
    tokenizer: AutoTokenizer,
    spec: TaskSpec,
    generate_n: int,
) -> tuple[pd.DataFrame, dict]:
    reconstructed_rows = []
    keep_count = 0
    blank_count = 0

    for i in range(len(input_df)):
        row = input_df.iloc[i]
        ground_truth_sid = extract_sid(str(row["groundtruth"]))
        predicted_sids: list[str] = []

        for j in range(generate_n):
            output = sid_outputs[i * generate_n + j]
            for completion in output.sequences:
                tokens = completion.tokens[-4:]
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                predicted = f"<|sid_begin|>{text}<|sid_end|>"
                extracted = extract_sid(predicted)
                if extracted:
                    predicted_sids.append(extracted)

        row_dict = row.to_dict()
        if ground_truth_sid and ground_truth_sid in predicted_sids:
            row_dict["recon_kept"] = True
            keep_count += 1
        else:
            row_dict = spec.apply_miss_policy(row_dict)
            blank_count += 1
        reconstructed_rows.append(row_dict)

    out_df = pd.DataFrame(reconstructed_rows)
    validate_output_alignment(input_df.reset_index(drop=True), out_df.reset_index(drop=True), spec)
    report = {
        "task": spec.task,
        "n": int(len(out_df)),
        "kept": int(keep_count),
        "blanked": int(blank_count),
        "keep_rate": round(keep_count / len(out_df), 6) if len(out_df) else 0.0,
    }
    return out_df, report


def reconstruct_shard(
    *,
    task: str,
    gpu_id: int,
    model_path: str,
    data_path: str,
    output_file: str,
    report_file: str,
    sample_size: Optional[int] = None,
    generate_n: int = 2,
    beam_width: int = 5,
    think_max_tokens: int = 100,
    sid_max_tokens: int = 4,
) -> None:
    spec = get_task_spec(task)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[GPU {gpu_id}] task={task} | CUDA_VISIBLE_DEVICES={visible}")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    print(f"[GPU {gpu_id}] 加载模型: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sampling_params_think = SamplingParams(
        n=generate_n,
        temperature=0.8,
        top_p=0.95,
        top_k=200,
        max_tokens=think_max_tokens,
        stop=["<|sid_begin|>"],
    )
    beam_params = BeamSearchParams(beam_width=beam_width, max_tokens=sid_max_tokens)

    input_df = pd.read_parquet(data_path)
    if sample_size:
        input_df = input_df.head(sample_size).reset_index(drop=True)
    validate_input_frame(input_df, spec)
    total_count = len(input_df)
    print(f"[GPU {gpu_id}] 样本数: {total_count}")

    prompts = [build_prompt(spec, str(row["description"])) for _, row in input_df.iterrows()]

    print(f"[GPU {gpu_id}] Step 1: Think 生成 (n={generate_n})...")
    think_outputs = llm.generate(prompts, sampling_params_think)

    sid_prompts: list[str] = []
    for output in think_outputs:
        for completion in output.outputs:
            sid_prompts.append(output.prompt + completion.text + "<|sid_begin|>")

    print(f"[GPU {gpu_id}] Step 2: SID Beam (width={beam_width}, {len(sid_prompts)} prompts)...")
    sid_outputs = llm.beam_search(prompts=sid_prompts, params=beam_params)

    print(f"[GPU {gpu_id}] Step 3: 重构数据...")
    out_df, report = reconstruct_records(
        input_df=input_df,
        sid_outputs=sid_outputs,
        tokenizer=tokenizer,
        spec=spec,
        generate_n=generate_n,
    )
    report.update({
        "gpu_id": gpu_id,
        "generate_n": generate_n,
        "beam_width": beam_width,
        "think_max_tokens": think_max_tokens,
        "sid_max_tokens": sid_max_tokens,
    })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_file, index=False)
    Path(report_file).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[GPU {gpu_id}] 完成 | keep-rate={report['keep_rate']:.2%} "
        f"({report['kept']}/{report['n']})"
    )

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_shard_subprocess(
    *,
    task: str,
    gpu_id: int,
    model_path: str,
    shard_path: str,
    output_file: str,
    report_file: str,
    generate_n: int,
    beam_width: int,
    think_max_tokens: int,
    sid_max_tokens: int,
    log_file: str,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--task", task,
        "--model_path", model_path,
        "--shard_mode",
        "--shard_path", shard_path,
        "--shard_output", output_file,
        "--shard_report", report_file,
        "--shard_gpu", "0",
        "--generate_n", str(generate_n),
        "--beam_width", str(beam_width),
        "--think_max_tokens", str(think_max_tokens),
        "--sid_max_tokens", str(sid_max_tokens),
    ]

    f = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc, f


def _parse_pipeline_category(config_name: str) -> str:
    if not config_name or "_" not in config_name:
        raise ValueError("config_name 必须形如 <Category>_<task>")
    return config_name.rsplit("_", 1)[0]


def resolve_output_paths(task: str, args: argparse.Namespace) -> tuple[str, str]:
    is_pipeline = args.config_name is not None and args.epoch is not None
    if is_pipeline:
        category = _parse_pipeline_category(args.config_name)
        out_dir = stage3_reconstruct_dir(category, task, args.epoch)
        return str(out_dir / "reconstructed_data.parquet"), str(out_dir / "reconstruction_report.json")
    if args.output_path is None:
        raise SystemExit("手动模式下 --output_path 为必填参数")
    output = Path(args.output_path)
    return str(output), str(output.with_name("reconstruction_report.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 统一重构脚本 (vLLM)")
    parser.add_argument("--task", type=str, default=None, help="ra 或 routing")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--config_name", type=str, default=None, help="流水线模式: 如 Beauty_ra / Beauty_routing")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--generate_n", type=int, default=2)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--think_max_tokens", type=int, default=100)
    parser.add_argument("--sid_max_tokens", type=int, default=4)

    parser.add_argument("--shard_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_report", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard_gpu", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    task = normalize_task(args.task, args.config_name)

    if args.shard_mode:
        if not args.shard_path or not args.shard_output or not args.shard_report:
            raise SystemExit("shard_mode requires --shard_path --shard_output --shard_report")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        reconstruct_shard(
            task=task,
            gpu_id=0,
            model_path=args.model_path,
            data_path=args.shard_path,
            output_file=args.shard_output,
            report_file=args.shard_report,
            sample_size=args.sample_size,
            generate_n=args.generate_n,
            beam_width=args.beam_width,
            think_max_tokens=args.think_max_tokens,
            sid_max_tokens=args.sid_max_tokens,
        )
        return

    if args.data_path is None:
        parser.error("--data_path is required (except in internal --shard_mode)")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    final_output, final_report = resolve_output_paths(task, args)

    print(f"task={task}")
    print(f"输出: {final_output}")
    print(
        f"参数: generate_n={args.generate_n}, beam_width={args.beam_width}, "
        f"think_max={args.think_max_tokens}, sid_max={args.sid_max_tokens}"
    )

    if args.num_gpus > 1:
        input_df = pd.read_parquet(args.data_path)
        if args.sample_size:
            input_df = input_df.head(args.sample_size).reset_index(drop=True)
        spec = get_task_spec(task)
        validate_input_frame(input_df, spec)

        total = len(input_df)
        worker_count = min(args.num_gpus, total)
        if worker_count < 1:
            raise SystemExit("输入数据为空，无法启动多卡重构")
        per_worker = math.ceil(total / worker_count)

        print(f"\n多卡模式 | GPUs={worker_count}")
        print(f"总样本: {total} | 每 GPU: ~{per_worker}")

        temp_dir = Path(final_output).parent / "temp_shards"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        shard_paths = []
        for i in range(worker_count):
            start = i * per_worker
            end = min(start + per_worker, total)
            shard_df = input_df.iloc[start:end].reset_index(drop=True)
            shard_path = temp_dir / f"input_shard_{i}.parquet"
            shard_df.to_parquet(shard_path, index=False)
            shard_paths.append(str(shard_path))
            print(f"  Shard {i}: [{start}:{end}] ({len(shard_df)} 条)")

        procs = []
        print(f"\n启动 {worker_count} 个并行 vLLM 子进程...")
        for i in range(worker_count):
            shard_out = str(temp_dir / f"output_shard_{i}.parquet")
            shard_report = str(temp_dir / f"report_gpu_{i}.json")
            log_f = str(temp_dir / f"log_gpu_{i}.txt")
            proc, f = run_shard_subprocess(
                task=task,
                gpu_id=i,
                model_path=args.model_path,
                shard_path=shard_paths[i],
                output_file=shard_out,
                report_file=shard_report,
                generate_n=args.generate_n,
                beam_width=args.beam_width,
                think_max_tokens=args.think_max_tokens,
                sid_max_tokens=args.sid_max_tokens,
                log_file=log_f,
            )
            procs.append((i, proc, f, shard_out, shard_report, log_f))
            time.sleep(2)

        failed, results, reports = [], [], []
        for gid, proc, f, out, report, log_f in procs:
            ret = proc.wait()
            f.close()
            if ret == 0:
                results.append((gid, out))
                reports.append((gid, report))
                print(f"[GPU {gid}] ✅ 完成")
            else:
                failed.append(gid)
                print(f"[GPU {gid}] ❌ 失败: Exit code {ret}")
                print(f"  日志: {log_f}")

        if failed:
            raise SystemExit(f"多卡重构失败, GPUs={failed}")

        print("\n合并 shard 输出...")
        results.sort(key=lambda x: x[0])
        merged = pd.concat([pd.read_parquet(out_path) for _, out_path in results], ignore_index=True)
        validate_output_alignment(input_df.reset_index(drop=True), merged.reset_index(drop=True), spec)
        Path(final_output).parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(final_output, index=False)

        reports.sort(key=lambda x: x[0])
        per_gpu_reports = [json.loads(Path(p).read_text(encoding="utf-8")) for _, p in reports]
        agg = {
            "task": task,
            "n": int(len(merged)),
            "per_gpu": per_gpu_reports,
            "kept": int(sum(item["kept"] for item in per_gpu_reports)),
            "blanked": int(sum(item["blanked"] for item in per_gpu_reports)),
            "generate_n": args.generate_n,
            "beam_width": args.beam_width,
            "think_max_tokens": args.think_max_tokens,
            "sid_max_tokens": args.sid_max_tokens,
        }
        agg["keep_rate"] = round(agg["kept"] / agg["n"], 6) if agg["n"] else 0.0
        Path(final_report).write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

        shutil.rmtree(temp_dir)
        print(f"✅ 多卡重构完成 → {final_output}")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    reconstruct_shard(
        task=task,
        gpu_id=args.gpu_id,
        model_path=args.model_path,
        data_path=args.data_path,
        output_file=final_output,
        report_file=final_report,
        sample_size=args.sample_size,
        generate_n=args.generate_n,
        beam_width=args.beam_width,
        think_max_tokens=args.think_max_tokens,
        sid_max_tokens=args.sid_max_tokens,
    )
    print(f"✅ 单卡重构完成 → {final_output}")


if __name__ == "__main__":
    main()
