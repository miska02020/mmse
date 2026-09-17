"""
读 models.csv，并发调用 other_batch.py 跑每个模型。

models.csv 列：model_name, api_key, url

用法：
  python run_all.py                          # 全部模型并发，每个 base_url 默认并发 2
  python run_all.py --default-cap 3          # 每个 base_url 默认最多同时跑 3 个
  python run_all.py --max-workers 4          # 全局线程池上限 4
  python run_all.py --models gpt-5.4-high kimi-k2.5  # 只跑指定的几个

  # 为特定 base_url 单独设置并发容量（可多次使用）：
  python run_all.py --url-cap "sg.uiuiapi.com=2" --url-cap "aiapi.world=1"
"""
import argparse
import csv
import os
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse


MODELS_CSV = "models.csv"
WORKER_SCRIPT = "run_img_batch.py.py"
LOG_DIR = "logs"

# print 在多线程里不是原子的，加个锁避免行内混杂
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


def extract_host(url: str) -> str:
    """从 base_url 中提取 host，用作分组键。"""
    parsed = urlparse(url)
    return parsed.hostname or url


def build_semaphores(models, url_caps: dict, default_cap: int):
    """为每个 base_url 创建一个 Semaphore，返回 {host: Semaphore}。"""
    hosts = set(extract_host(m["url"]) for m in models)
    sems = {}
    for h in sorted(hosts):
        cap = url_caps.get(h, default_cap)
        sems[h] = threading.Semaphore(cap)
        safe_print(f"  base_url [{h}] 并发上限 = {cap}")
    return sems


def run_one(model_name, api_key, url, semaphores):
    """跑一个模型。用对应 host 的信号量控制并发。返回 (model_name, returncode)。"""
    host = extract_host(url)
    sem = semaphores[host]

    safe_print(f"[{model_name}] 等待 {host} 的并发名额...")
    with sem:
        safe_print(f"[{model_name}] 获得名额，启动")
        return _do_run(model_name, api_key, url)


def _do_run(model_name, api_key, url):
    """实际执行子进程。"""
    log_path = os.path.join(LOG_DIR, f"{model_name}.log")
    cmd = [sys.executable, WORKER_SCRIPT, model_name, api_key, url]

    safe_print(f"[{model_name}] 启动 -> 日志 {log_path}")

    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n========== {datetime.now().isoformat()} 启动 ==========\n")
            logf.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in proc.stdout:
                line = line.rstrip("\n")
                logf.write(line + "\n")
                logf.flush()
                safe_print(f"[{model_name}] {line}")

            rc = proc.wait()

        if rc == 0:
            safe_print(f"[{model_name}] ✅ 完成")
        else:
            safe_print(f"[{model_name}] ❌ 退出码 {rc}")
        return model_name, rc

    except Exception as e:
        safe_print(f"[{model_name}] 💥 异常: {type(e).__name__}: {e}")
        return model_name, -1


def load_models(path, only=None):
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("model_name", "").strip()]
    if only:
        wanted = set(only)
        rows = [r for r in rows if r["model_name"] in wanted]
        missing = wanted - {r["model_name"] for r in rows}
        if missing:
            safe_print(f"警告：在 {path} 里没找到这些模型：{missing}")
    return rows


def parse_url_caps(raw_list):
    """解析 --url-cap 参数，如 ["sg.uiuiapi.com=2", "aiapi.world=1"]。"""
    caps = {}
    if not raw_list:
        return caps
    for item in raw_list:
        if "=" not in item:
            safe_print(f"警告：--url-cap 格式错误（应为 host=数字）：{item}")
            continue
        host, val = item.rsplit("=", 1)
        try:
            caps[host.strip()] = int(val.strip())
        except ValueError:
            safe_print(f"警告：--url-cap 值不是整数：{item}")
    return caps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-csv", default=MODELS_CSV)
    parser.add_argument("--max-workers", type=int, default=None,
                        help="全局线程池上限，默认 = 模型数（全部并发）")
    parser.add_argument("--default-cap", type=int, default=9,
                        help="每个 base_url 的默认并发上限（默认 2）")
    parser.add_argument("--url-cap", action="append", default=None,
                        help="为指定 host 设置并发上限，格式: host=数字，可多次使用。"
                             "例: --url-cap sg.uiuiapi.com=2 --url-cap aiapi.world=1")
    parser.add_argument("--models", nargs="*", help="只跑指定的几个 model_name")
    args = parser.parse_args()

    if not os.path.exists(WORKER_SCRIPT):
        safe_print(f"找不到 {WORKER_SCRIPT}，请放在同一目录下。")
        sys.exit(1)

    models = load_models(args.models_csv, only=args.models)
    if not models:
        safe_print("没有要跑的模型，退出。")
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)

    # 统计每个 host 下有多少模型
    host_models = defaultdict(list)
    for m in models:
        host_models[extract_host(m["url"])].append(m["model_name"])

    safe_print(f"共 {len(models)} 个模型，分布在 {len(host_models)} 个 base_url：")
    for h, names in host_models.items():
        safe_print(f"  {h}: {names}")
    safe_print("")

    # 构建信号量
    url_caps = parse_url_caps(args.url_cap)
    safe_print("并发容量配置：")
    semaphores = build_semaphores(models, url_caps, args.default_cap)
    safe_print("")

    workers = args.max_workers or len(models)
    safe_print(f"全局线程池大小 = {workers}\n")

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, m["model_name"], m["api_key"], m["url"], semaphores): m["model_name"]
            for m in models
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, rc = fut.result()
                results[name] = rc
            except Exception as e:
                safe_print(f"[{name}] 💥 future 异常: {e}")
                results[name] = -1

    # 汇总
    safe_print("\n========== 全部结束 ==========")
    ok = [n for n, rc in results.items() if rc == 0]
    bad = [(n, rc) for n, rc in results.items() if rc != 0]
    safe_print(f"成功 {len(ok)} 个：{ok}")
    if bad:
        safe_print(f"失败 {len(bad)} 个：")
        for n, rc in bad:
            safe_print(f"  - {n} (退出码 {rc}, 看 {LOG_DIR}/{n}.log)")


if __name__ == "__main__":
    main()
