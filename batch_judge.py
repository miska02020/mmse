#!/usr/bin/env python3
"""
自动统计 output_*.csv 数量，从 check_key.csv 中选择对应行数的 keys，
批量运行 judge_other.py 进行评估。
支持 Key 池复用：如果 Key 数量少于模型数量，前面的模型跑完释放 Key 后，后面的模型排队使用。
"""
import glob
import csv
import subprocess
import sys
import os
import re
import queue  # <--- 新增：用于管理 Key 资源池
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

KEYS_CSV = "check_key.csv"
JUDGE_SCRIPT = "judge_other.py"
LOG_DIR = "judge_logs"

# 线程安全的打印
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)

def extract_model_name(output_path):
    """从 output_xxx.csv 中提取模型名"""
    base = os.path.basename(output_path)
    m = re.match(r"output_(.+)\.csv$", base)
    return m.group(1) if m else None

def load_keys(csv_path, max_count=None):
    """从 check_key.csv 中读取 key 行"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        keys_rows = []
        for i, row in enumerate(reader):
            if max_count is not None and i >= max_count:
                break
            if len(row) >= 3:
                keys_rows.append({
                    'api_key_1': row[0].strip(),
                    'api_key_2': row[1].strip(),
                    'api_key_3': row[2].strip()
                })
        return keys_rows

def run_judge(model_name, key_queue):
    """运行单个模型的 judge_other.py，从队列获取 key，用完归还"""
    # 阻塞等待，直到有可用的 key (如果 Key 不够，后面的模型会在这里排队等待)
    keys = key_queue.get()
    
    log_path = os.path.join(LOG_DIR, f"judge_{model_name}.log")
    cmd = [
        sys.executable, JUDGE_SCRIPT, model_name,
        keys['api_key_1'], keys['api_key_2'], keys['api_key_3']
    ]
    safe_print(f"[{model_name}] 🔑 获取到 Key，启动评估 -> 日志 {log_path}")
    
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"命令: {' '.join(cmd)}\n")
            logf.write("=" * 50 + "\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1
            )
            for line in proc.stdout:
                line = line.rstrip()
                logf.write(line + "\n")
                logf.flush()
                # safe_print(f"[{model_name}] {line}") # 如果控制台输出太乱，可以注释掉这行
            rc = proc.wait()
            if rc == 0:
                safe_print(f"[{model_name}] ✅ 评估完成")
            else:
                safe_print(f"[{model_name}] ❌ 退出码 {rc}")
            return model_name, rc
    except Exception as e:
        safe_print(f"[{model_name}] 💥 异常: {e}")
        return model_name, -1
    finally:
        # 【核心】无论成功还是失败，都必须将 key 归还到队列，供后面的模型使用
        key_queue.put(keys)
        safe_print(f"[{model_name}] 🔓 Key 已归还队列")

def main():
    # 统计 output_*.csv 数量
    output_files = glob.glob("output_*.csv")
    if not output_files:
        safe_print("当前目录没有 output_*.csv 文件。")
        return
        
    safe_print(f"发现 {len(output_files)} 个 output CSV 文件：")
    models = []
    for f in sorted(output_files):
        model = extract_model_name(f)
        if model:
            models.append(model)
            safe_print(f"  - {f} -> 模型: {model}")
        else:
            safe_print(f"  - {f} -> ⚠️  无法提取模型名")
            
    if not models:
        safe_print("没有有效的模型文件，退出。")
        return
        
    safe_print(f"\n需要评估 {len(models)} 个模型")
    
    # 检查必要文件
    if not os.path.exists(KEYS_CSV):
        safe_print(f"错误：找不到 {KEYS_CSV}")
        return
    if not os.path.exists(JUDGE_SCRIPT):
        safe_print(f"错误：找不到 {JUDGE_SCRIPT}")
        return
        
    # 读取所有的 keys
    safe_print(f"\n从 {KEYS_CSV} 中读取所有可用 keys...")
    try:
        keys_list = load_keys(KEYS_CSV) # 不再限制数量，读取所有
    except Exception as e:
        safe_print(f"读取 keys 失败: {e}")
        return
        
    if not keys_list:
        safe_print("错误：没有读取到任何 key，退出。")
        return
        
    safe_print(f"共找到 {len(keys_list)} 组 Key。")
    
    if len(keys_list) < len(models):
        safe_print(f"⚠️ 警告：Key 数量 ({len(keys_list)}) 少于模型数量 ({len(models)})。")
        safe_print("💡 将启用 Key 池复用模式：前面的模型跑完释放 Key 后，后面的模型排队使用。")
        
    # 初始化 Key 队列 (线程安全的资源池)
    key_queue = queue.Queue()
    for k in keys_list:
        key_queue.put(k)
        
    # 准备日志目录
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 并发运行评估
    safe_print(f"\n开始评估 {len(models)} 个模型 (实际最大并发数受限于 Key 数量)...\n")
    results = {}
    
    # max_workers 设置为模型数量，由于 key_queue.get() 会阻塞，
    # 实际同时运行的任务数永远不会超过 key_queue 中的 Key 数量。
    max_workers = len(models)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for model in models:
            fut = executor.submit(run_judge, model, key_queue)
            futures[fut] = model
            
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                _, rc = fut.result()
                results[model] = rc
            except Exception as e:
                safe_print(f"[{model}] 💥 future 异常: {e}")
                results[model] = -1
                
    # 汇总结果
    safe_print("\n" + "=" * 60)
    safe_print("评估完成汇总：")
    success = [m for m, rc in results.items() if rc == 0]
    failed = [(m, rc) for m, rc in results.items() if rc != 0]
    safe_print(f"✅ 成功 {len(success)} 个：{success}")
    if failed:
        safe_print(f"❌ 失败 {len(failed)} 个：")
        for m, rc in failed:
            safe_print(f"   - {m} (退出码 {rc}, 日志: {LOG_DIR}/judge_{m}.log)")

if __name__ == "__main__":
    main()