#!/usr/bin/env python3
"""
自动统计 output_*.csv 数量，从 check_key.csv 中选择对应行数的 keys，
批量运行 judge_other.py 进行评估。
如果 keys 数量不足，则循环复用。
"""
import glob
import csv
import subprocess
import sys
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

KEYS_CSV = "check_key1.csv"
JUDGE_SCRIPT = "judge_other_yue.py"
LOG_DIR = "judge_logs_yue"

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


def load_all_keys(csv_path):
    """读取 check_key.csv 中所有有效 key 行（每行至少三列）"""
    keys_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # 跳过表头
        except StopIteration:
            return keys_rows
        for row in reader:
            if len(row) >= 3:
                keys_rows.append({
                    'api_key_1': row[0].strip(),
                    'api_key_2': row[1].strip(),
                    'api_key_3': row[2].strip()
                })
    return keys_rows


def run_judge(model_name, keys, key_index):
    """运行单个模型的 judge_other.py，并记录使用的 key 组索引"""
    log_path = os.path.join(LOG_DIR, f"judge_{model_name}.log")
    cmd = [
        sys.executable, JUDGE_SCRIPT, model_name,
        keys['api_key_1'], keys['api_key_2'], keys['api_key_3']
    ]
    
    safe_print(f"[{model_name}] 启动评估 (使用 keys 组 #{key_index+1}) -> 日志 {log_path}")
    
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"命令: {' '.join(cmd)}\n")
            logf.write(f"使用的 key 组索引: {key_index+1}\n")
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
                safe_print(f"[{model_name}] {line}")
            
            rc = proc.wait()
        
        if rc == 0:
            safe_print(f"[{model_name}] ✅ 评估完成")
        else:
            safe_print(f"[{model_name}] ❌ 退出码 {rc}")
        return model_name, rc
        
    except Exception as e:
        safe_print(f"[{model_name}] 💥 异常: {e}")
        return model_name, -1


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
    
    # 读取所有 key 组
    safe_print(f"\n从 {KEYS_CSV} 中读取所有 keys...")
    try:
        all_keys = load_all_keys(KEYS_CSV)
    except Exception as e:
        safe_print(f"读取 keys 失败: {e}")
        return
        
    if not all_keys:
        safe_print(f"错误：{KEYS_CSV} 中没有有效的 key 行（至少三列）")
        return
        
    key_count = len(all_keys)
    safe_print(f"共读取 {key_count} 组 keys")
    if key_count < len(models):
        safe_print(f"⚠️  key 组数 ({key_count}) 少于模型数 ({len(models)})，将循环复用 keys")
    else:
        safe_print(f"✅ key 组数足够，每个模型使用独立的一组")
    
    # 准备日志目录
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 并发运行评估
    safe_print(f"\n开始并发评估 {len(models)} 个模型...\n")
    results = {}
    
    # 可以调整 max_workers，避免过载
    max_workers = len(models)  
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, model in enumerate(models):
            keys = all_keys[i % key_count]   # 循环分配
            fut = executor.submit(run_judge, model, keys, i % key_count)
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