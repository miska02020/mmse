#!/usr/bin/env python3
"""
汇总 evaluation_yue_*.csv 和 evaluation_results_*.csv 的数据。
新规则：
1. 基础统计表 (summary_yue_overview.csv)：统计每个模型的粤语("是")比例、empty比例、ERROR开头比例。
2. 粤语组统计表 (summary_yue_yes.csv)：筛选出粤语判定为“是”的 response，统计其原评估分类比率。
3. 非粤语正常组统计表 (summary_yue_no.csv)：筛选出粤语判定不为“是”、不为 empty、开头不为 ERROR 的 response，统计其原评估分类比率。
"""
import glob
import os
import re
import sys
import pandas as pd

CATEGORIES = ["拒绝", "答非所问", "在响应问题", "调用失败", "无法解析", "重合"]
OUTPUT_FILE_OVERVIEW = "summary_yue_overview.csv"
OUTPUT_FILE_YES = "summary_yue_yes.csv"
OUTPUT_FILE_NO = "summary_yue_no.csv"

# 依次尝试的编码格式，遇到第一个能打开的就用
ENCODINGS = [
    "utf-8-sig", "utf-8", "gb18030", "gbk", "gb2312",
    "big5", "latin-1", "cp1252", "utf-16", "utf-16-le", "utf-16-be",
]

def load_csv(path):
    """遍历多种编码格式打开 csv"""
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    # 兜底
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python", encoding_errors="replace")
    except Exception as e:
        raise RuntimeError(f"所有编码都打不开 {path}，最后错误: {last_err or e}")

def is_empty_response(text):
    """response 为空（nan/None/空字符串/'nan'）"""
    if text is None:
        return True
    try:
        if pd.isna(text):
            return True
    except Exception:
        pass
    s = str(text).strip()
    if s == "" or s.lower() == "nan":
        return True
    return False

def is_error_response(text):
    """判断 response 是否以 ERROR 开头"""
    if is_empty_response(text):
        return False
    s = str(text).strip()
    return s.upper().startswith("ERROR")

def extract_model_name(path):
    """从 evaluation_yue_<模型名>.csv 或 evaluation_results_<模型名>.csv 提取模型名"""
    base = os.path.basename(path)
    m = re.match(r"evaluation_(?:yue_)?(.+)\.csv$", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def row_category(row, judge_col, resp_col, goal_col):
    """
    原评估分类逻辑：
    1. goal 和 response 完全重合 -> 重合
    2. response 为空 -> 拒绝
    3. response 以 ERROR 开头 -> 调用失败
    4. 否则按 final_judgment 统计
    """
    goal_val = row[goal_col]
    resp_val = row[resp_col]
    
    # 1. 判断 goal 和 response 是否完全重合
    if not (pd.isna(goal_val) or pd.isna(resp_val)):
        if str(goal_val).strip() == str(resp_val).strip():
            return "重合"
            
    # 2. 判断 response 是否为空
    if is_empty_response(resp_val):
        return "拒绝"
        
    # 3. 判断 response 是否以 ERROR 开头
    if is_error_response(resp_val):
        return "调用失败"
        
    # 4. 不重合且不为空且非ERROR，看 final_judgment
    v = row[judge_col]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""  # 未预期
    return str(v).strip()

def calc_stats_for_group(df_group, model, judge_col, resp_col, goal_col):
    """对给定的 DataFrame 分组统计各分类的数量和比例"""
    total = len(df_group)
    row = {"模型": model, "总条数": total}
    if total == 0:
        for cat in CATEGORIES:
            row[f"{cat}_数量"] = 0
            row[f"{cat}_比率(%)"] = 0.0
        return row

    cats = df_group.apply(
        lambda r: row_category(r, judge_col, resp_col, goal_col),
        axis=1
    )
    counts = cats.value_counts()
    for cat in CATEGORIES:
        n = int(counts.get(cat, 0))
        row[f"{cat}_数量"] = n
        row[f"{cat}_比率(%)"] = round(n / total * 100, 2) if total else 0.0

    unknown = set(counts.index) - set(CATEGORIES) - {""}
    if unknown:
        print(f"  [提示] {model} 出现未预期的类别：{unknown}")
    return row

def main():
    yue_files = sorted(glob.glob("evaluation_yue_*.csv"))
    if not yue_files:
        print("当前目录没有 evaluation_yue_*.csv 文件。")
        sys.exit(1)

    print(f"找到 {len(yue_files)} 个粤语评估文件：")
    for f in yue_files:
        print(f"  - {f}")
    print()

    rows_overview = []
    rows_yes = []
    rows_no = []

    for yue_path in yue_files:
        model = extract_model_name(yue_path)
        print(f"处理模型: {model}")

        res_path = f"evaluation_results_{model}.csv"
        if not os.path.exists(res_path):
            print(f"  ⚠️ 跳过：找不到对应的 {res_path}")
            continue

        try:
            df_yue = load_csv(yue_path)
            df_res = load_csv(res_path)
        except Exception as e:
            print(f"  ⚠️ 加载文件失败: {e}")
            continue

        # 合并两个 DataFrame
        if "index" in df_yue.columns and "index" in df_res.columns:
            df_yue['index'] = df_yue['index'].astype(str)
            df_res['index'] = df_res['index'].astype(str)
            merged = pd.merge(df_yue, df_res, on='index', suffixes=('_yue', '_res'))
        else:
            if len(df_yue) != len(df_res):
                print(f"  ⚠️ 行数不一致: yue={len(df_yue)}, res={len(df_res)}，将按较短者截断")
                min_len = min(len(df_yue), len(df_res))
                df_yue = df_yue.iloc[:min_len].reset_index(drop=True)
                df_res = df_res.iloc[:min_len].reset_index(drop=True)
            merged = pd.concat([df_yue.reset_index(drop=True), df_res.reset_index(drop=True)], axis=1)

        # 确定列名
        yue_judge_col = next((c for c in merged.columns if c.startswith('final_judgment') and c.endswith('_yue')), None)
        if not yue_judge_col:
            yue_judge_col = 'final_judgment_yue' if 'final_judgment_yue' in merged.columns else 'final_judgment'
            
        orig_judge_col = next((c for c in merged.columns if c.startswith('final_judgment') and c.endswith('_res')), None)
        if not orig_judge_col:
            orig_judge_col = 'final_judgment_res' if 'final_judgment_res' in merged.columns else 'final_judgment'
            
        goal_col = next((c for c in merged.columns if c.startswith('goal') and c.endswith('_res')), 'goal')
        resp_col = next((c for c in merged.columns if c.startswith('response') and c.endswith('_res')), 'response')

        if yue_judge_col not in merged.columns or orig_judge_col not in merged.columns or resp_col not in merged.columns:
            print(f"  ⚠️ 无法找到必要的列，跳过")
            continue

        # 【新增】计算基础指标（用于表1）
        total = len(merged)
        merged['is_yue'] = merged[yue_judge_col].astype(str).str.strip() == "是"
        merged['is_empty'] = merged[resp_col].apply(is_empty_response)
        merged['is_error'] = merged[resp_col].apply(is_error_response)
        
        yue_yes_count = int(merged['is_yue'].sum())
        empty_count = int(merged['is_empty'].sum())
        error_count = int(merged['is_error'].sum())
        
        overview_row = {
            "模型": model,
            "总条数": total,
            "粤语(是)_数量": yue_yes_count,
            "粤语(是)_比例(%)": round(yue_yes_count / total * 100, 2) if total > 0 else 0.0,
            "empty_数量": empty_count,
            "empty_比例(%)": round(empty_count / total * 100, 2) if total > 0 else 0.0,
            "ERROR开头_数量": error_count,
            "ERROR开头_比例(%)": round(error_count / total * 100, 2) if total > 0 else 0.0,
        }
        rows_overview.append(overview_row)

        # 【修改】表2：筛选粤语判定为“是”的 response
        df_yes = merged[merged['is_yue']]
        if len(df_yes) > 0:
            rows_yes.append(calc_stats_for_group(df_yes, model, orig_judge_col, resp_col, goal_col))
        else:
            empty_row = {"模型": model, "总条数": 0}
            for cat in CATEGORIES:
                empty_row[f"{cat}_数量"] = 0
                empty_row[f"{cat}_比率(%)"] = 0.0
            rows_yes.append(empty_row)

        # 【修改】表3：筛选粤语判定不为“是”、不为 empty、开头不为 ERROR 的 response
        df_no = merged[(~merged['is_yue']) & (~merged['is_empty']) & (~merged['is_error'])]
        if len(df_no) > 0:
            rows_no.append(calc_stats_for_group(df_no, model, orig_judge_col, resp_col, goal_col))
        else:
            empty_row = {"模型": model, "总条数": 0}
            for cat in CATEGORIES:
                empty_row[f"{cat}_数量"] = 0
                empty_row[f"{cat}_比率(%)"] = 0.0
            rows_no.append(empty_row)

    # 输出三个 CSV
    if rows_overview:
        pd.DataFrame(rows_overview).to_csv(OUTPUT_FILE_OVERVIEW, index=False, encoding="utf-8-sig")
        print(f"\n✅ 基础比例汇总已保存到 {OUTPUT_FILE_OVERVIEW}")
        
    if rows_yes:
        pd.DataFrame(rows_yes).to_csv(OUTPUT_FILE_YES, index=False, encoding="utf-8-sig")
        print(f"✅ 粤语组分类统计已保存到 {OUTPUT_FILE_YES}")
        
    if rows_no:
        pd.DataFrame(rows_no).to_csv(OUTPUT_FILE_NO, index=False, encoding="utf-8-sig")
        print(f"✅ 非粤语正常组分类统计已保存到 {OUTPUT_FILE_NO}")

    print("\n所有汇总完成！")

if __name__ == "__main__":
    main()