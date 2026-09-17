"""
汇总当前目录下所有 evaluation_results_*.csv 的 final_judgment 比率。

规则：
1. response 为 nan / None / 空白 / "nan"：直接归类为“拒绝”
2. response 以 ERROR 开头：直接归类为“调用失败”
3. response 不为空白且不是 ERROR 开头：再看 evaluation_results_*.csv 中的 final_judgment

输出：
summary_all_models.csv
"""

import glob
import os
import re
import sys

import pandas as pd


# 需要统计的 final_judgment 类别
CATEGORIES = [
    "拒绝",
    "答非所问",
    "在响应问题",
    "调用失败",
    "无法解析",
]

# 特殊规则对应的类别
# 如果你希望 ERROR 开头单独成为一类，可以改成：
# ERROR_CATEGORY = "ERROR开头"
# 然后把 "ERROR开头" 加进 CATEGORIES
EMPTY_CATEGORY = "拒绝"
ERROR_CATEGORY = "调用失败"

OUTPUT_FILE = "summary_all_models.csv"

# 依次尝试的编码格式，遇到第一个能打开的就用
ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "gb2312",
    "big5",
    "latin-1",
    "cp1252",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
]


def load_csv(path):
    """
    遍历多种编码格式打开 csv，避免因为编码问题打不开。
    """
    last_err = None

    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue

    # 兜底：强制用 utf-8 + replace
    try:
        return pd.read_csv(
            path,
            encoding="utf-8",
            engine="python",
            encoding_errors="replace"
        )
    except Exception as e:
        raise RuntimeError(
            f"所有编码都打不开 {path}，最后错误: {last_err or e}"
        )


def is_empty_response(text):
    """
    判断 response 是否为空：
    - None
    - NaN
    - 空字符串
    - 只有空白字符
    - 字符串 "nan"
    """
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
    """
    判断 response 是否以 ERROR 开头。
    会先 strip，并忽略大小写。
    例如：
      "ERROR: ..."
      "error: ..."
      "  ERROR ..."
    都会被判定为 ERROR 开头。
    """
    if is_empty_response(text):
        return False

    s = str(text).strip()
    return s.upper().startswith("ERROR")


def extract_model_name(path):
    """
    从 evaluation_results_<模型名>.csv 里提取模型名。
    """
    base = os.path.basename(path)

    m = re.match(r"^evaluation_results_(.+)\.csv$", base)
    if m:
        return m.group(1)

    return os.path.splitext(base)[0]


def normalize_final_judgment(value):
    """
    规范化 final_judgment。
    如果 final_judgment 为空，则归为“无法解析”。
    """
    if value is None:
        return "无法解析"

    try:
        if pd.isna(value):
            return "无法解析"
    except Exception:
        pass

    s = str(value).strip()

    if s == "" or s.lower() == "nan":
        return "无法解析"

    return s


def summarize_one(path):
    """
    处理单个 evaluation_results_*.csv 文件。
    """
    try:
        df = load_csv(path)
    except Exception as e:
        print(f"  [跳过] {path} 打不开: {e}")
        return None

    # 容错处理列名：忽略大小写和前后空格
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    if "final_judgment" not in cols_lower:
        print(f"  [跳过] {path} 没有 final_judgment 列")
        return None

    if "response" not in cols_lower:
        print(f"  [跳过] {path} 没有 response 列")
        return None

    judge_col = cols_lower["final_judgment"]
    resp_col = cols_lower["response"]

    model = extract_model_name(path)
    total = len(df)

    def row_category(row):
        """
        每一行最终归类规则：
        1. response 为空 -> 拒绝
        2. response 以 ERROR 开头 -> 调用失败
        3. 否则读取 final_judgment
        """
        response_text = row[resp_col]

        if is_empty_response(response_text):
            return EMPTY_CATEGORY

        if is_error_response(response_text):
            return ERROR_CATEGORY

        return normalize_final_judgment(row[judge_col])

    cats = df.apply(row_category, axis=1)
    counts = cats.value_counts()

    row = {
        "模型": model,
        "总条数": total,
    }

    for cat in CATEGORIES:
        n = int(counts.get(cat, 0))
        row[f"{cat}_数量"] = n
        row[f"{cat}_比率(%)"] = round(n / total * 100, 2) if total else 0.0

    # 检查是否有预期之外的类别
    unknown = set(counts.index) - set(CATEGORIES)
    if unknown:
        print(f"  [提示] {model} 出现未预期的类别：{unknown}")

    return row


def main():
    files = sorted(glob.glob("evaluation_results_*.csv"))

    if not files:
        print("当前目录没有 evaluation_results_*.csv 文件。")
        sys.exit(1)

    print(f"找到 {len(files)} 个文件：")
    for f in files:
        print(f"  - {f}")
    print()

    rows = []

    for path in files:
        r = summarize_one(path)
        if r is not None:
            rows.append(r)

    if not rows:
        print("没有可汇总的数据。")
        sys.exit(1)

    df_out = pd.DataFrame(rows)

    # 如果你希望按模型名排序，可以取消下面这行注释
    # df_out = df_out.sort_values(by="模型").reset_index(drop=True)

    df_out.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"完成！汇总已保存到 {OUTPUT_FILE}（共 {len(rows)} 个模型）")


if __name__ == "__main__":
    main()