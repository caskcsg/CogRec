#!/usr/bin/env python3
"""
data/download_amazon_dataset.py — 原始数据下载器
=============================================
当前协议基于 Amazon2014 数据。

功能:
  --category  指定类目: Beauty / Sports / Toys / all
  --type      指定下载类型: review / meta / both
  --check     检查本地文件状态
  --dry-run   仅打印下载计划，不实际下载
  --force     强制覆盖已有文件
"""

import argparse
import ast
import gzip
import logging
import ssl
import time
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (  # type: ignore
    DATASETS,
    RAW_DIR,
    setup_logging,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def _progress(done: int, total: int, width: int = 40) -> None:
    """
    纯 ASCII 进度条，避免 Windows / GBK 控制台下的 UnicodeEncodeError
    """
    if total <= 0:
        try:
            print(f"\r  {_human(done)}", end="", flush=True)
        except UnicodeEncodeError:
            print(f"\r  {done} B", end="", flush=True)
        return

    frac = min(done / total, 1.0)
    n_full = int(width * frac)

    # 只使用 ASCII 字符
    bar = "#" * n_full + "-" * (width - n_full)

    msg = f"\r  [{bar}] {frac * 100:5.1f}%  {_human(done)} / {_human(total)}"
    try:
        print(msg, end="", flush=True)
    except UnicodeEncodeError:
        # 极端情况下退化为最简单输出
        print(f"\r  {frac * 100:5.1f}%  {done}/{total}", end="", flush=True)


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _category_cfg(category: str) -> dict:
    if category not in DATASETS:
        raise KeyError(f"未知类目: {category}")
    return DATASETS[category]


def _build_urls(category: str) -> dict:
    cfg = _category_cfg(category)
    return {
        "review": cfg["review"],
        "meta": cfg["meta"],
    }


def _dest(category: str, data_type: str) -> Path:
    cfg = _category_cfg(category)
    if data_type == "review":
        filename = cfg["review_filename"]
    elif data_type == "meta":
        filename = cfg["meta_filename"]
    else:
        raise ValueError(f"未知数据类型: {data_type}")
    return RAW_DIR / category / filename


def _head_content_length(url: str, timeout: int = 30) -> int | None:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def download_file(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (dataset-downloader/1.0)"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        log.info("  断点续传，已有 %s", _human(existing))

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
                remote_size = resp.headers.get("Content-Length")
                total = int(remote_size) + existing if remote_size is not None else 0
                mode = "ab" if existing > 0 else "wb"
                done = existing

                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(2 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        _progress(done, total)
            print()
            log.info("  完成 → %s (%s)", dest.name, _human(dest.stat().st_size))
            return dest

        except (URLError, HTTPError, OSError, ssl.SSLError) as e:
            print()
            log.warning("  第 %d/%d 次失败: %s", attempt, retries, e)
            if attempt < retries:
                wait = min(2 ** attempt, 60)
                log.info("  %ds 后重试 …", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"下载失败: {url}") from e

    return dest


def verify_gzip(path: Path, n: int = 5) -> bool:
    try:
        nonempty = 0
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nonempty += 1

                candidate = (
                    line.replace("true", "True")
                        .replace("false", "False")
                        .replace("null", "None")
                )
                try:
                    obj = ast.literal_eval(candidate)
                    if not isinstance(obj, dict):
                        return False
                except Exception:
                    pass

                if nonempty >= n:
                    break
        return nonempty > 0
    except Exception:
        return False


def download_dataset(
    category: str,
    data_type: str = "both",
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    urls = _build_urls(category)
    cfg = _category_cfg(category)
    types = ["review", "meta"] if data_type == "both" else [data_type]
    results = {}

    for t in types:
        url = urls[t]
        dest = _dest(category, t)
        size_hint = _head_content_length(url)

        if dry_run:
            log.info("[DRY-RUN] %s/%s", category, t)
            log.info("  来源: %s", url)
            log.info("  目标: %s", dest)
            if size_hint is not None:
                log.info("  预估大小: %s", _human(size_hint))
            results[t] = dest
            continue

        if dest.exists() and not force:
            log.info("已存在: %s (%s) — 跳过", dest.name, _human(dest.stat().st_size))
            results[t] = dest
            continue

        if force and dest.exists():
            dest.unlink()
            log.info("已删除: %s", dest.name)

        log.info("下载 %s / %s …", category, t)
        log.info("  label: %s", cfg["label"])
        log.info("  来源: %s", url)
        log.info("  目标: %s", dest)
        if size_hint is not None:
            log.info("  预估大小: %s", _human(size_hint))

        try:
            download_file(url, dest)
        except RuntimeError as e:
            log.error("%s", e)
            continue

        if verify_gzip(dest):
            log.info("✓ gzip 文件校验通过")
        else:
            log.warning("✗ 文件疑似损坏: %s", dest)

        results[t] = dest

    return results


def check_files() -> None:
    print(f"\n{'类别':<10} {'类型':<8} {'文件名':<48} {'大小':>14}  状态")
    print("─" * 92)
    for cat in DATASETS:
        cfg = _category_cfg(cat)
        for t in ("review", "meta"):
            dest = _dest(cat, t)
            size = _human(dest.stat().st_size) if dest.exists() else "(unknown)"
            status = "✓ 已下载" if dest.exists() else "✗ 未下载"
            fname = cfg["review_filename"] if t == "review" else cfg["meta_filename"]
            print(f"{cat:<10} {t:<8} {fname:<48} {size:>14}  {status}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="原始数据下载器")
    p.add_argument("--category", choices=list(DATASETS) + ["all"], default="all")
    p.add_argument("--type", choices=["review", "meta", "both"], default="both")
    p.add_argument("--check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.check:
        check_files()
        return

    cats = list(DATASETS) if args.category == "all" else [args.category]

    log.info("类别: %s", ", ".join(cats))
    log.info("类型: %s", args.type)

    t0 = time.time()
    for cat in cats:
        log.info("═" * 60)
        log.info("类别: %s", cat)
        download_dataset(
            category=cat,
            data_type=args.type,
            dry_run=args.dry_run,
            force=args.force,
        )
    log.info("═" * 60)
    log.info("完成，耗时 %.1fs", time.time() - t0)


if __name__ == "__main__":
    setup_logging("download_amazon_dataset")
    main()
