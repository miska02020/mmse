import json
import csv
import sys
import pandas as pd
import time
import os
from openai import OpenAI

# ============ 命令行参数 ============
if len(sys.argv) < 5:
    print("用法: python judge_other.py <模型名>")
    print("示例: python judge_other.py doubao")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
# ====================================

API_BASE_URL_1 = "https://aiapi.world/v1"  # 替换为你的中转站链接
API_KEY_1 = sys.argv[2]  # 替换为你的API Key
API_BASE_URL_2 = "https://aiapi.world/v1"  # 替换为你的中转站链接
API_KEY_2 = sys.argv[3]  # 替换为你的API Key
API_BASE_URL_3 = "https://aiapi.world/v1"  # 替换为你的中转站链接
API_KEY_3 = sys.argv[4]  # 替换为你的API Key

MODEL_1 = "gemini-3.1-flash-lite-preview"  # 替换为第一个判定模型名称
MODEL_2 = "gpt-5.4-mini"                    # 替换为第二个判定模型名称
MODEL_3 = "gpt-5.5"      # 替换为第三个仲裁模型名称

RESULTS_csv = f"evaluation_results_{MODEL_NAME}.csv"

# ============ 粤语特征词 ============
CANTONESE_MARKERS_CSV = "cantonese_markers.csv"

def load_cantonese_markers(path):
    """尝试多种编码打开粤语特征词 CSV，返回特征词集合。"""
    encodings = ["utf-8-sig", "utf-8", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            markers = set(df["特征词"].dropna().astype(str).tolist())
            print(f"粤语特征词加载成功（编码 {enc}），共 {len(markers)} 个词")
            return markers
        except (UnicodeDecodeError, KeyError):
            continue
    print("⚠️  粤语特征词文件加载失败，将全部走模型判断")
    return set()

CANTONESE_MARKERS = load_cantonese_markers(CANTONESE_MARKERS_CSV)

def has_cantonese(text: str) -> tuple[bool, str]:
    """检测 text 中是否含粤语特征词，返回 (是否含有, 命中的词)。"""
    for marker in CANTONESE_MARKERS:
        if marker in str(text):
            return True, marker
    return False, ""
# =====================================                    # 模型输出文件
OUTPUT_CSV = f"evaluation_yue_{MODEL_NAME}.csv"         # 输出结果文件

print(f"当前处理模型: {MODEL_NAME}")
print(f"  输入文件: {RESULTS_csv}")
print(f"  输出文件: {OUTPUT_CSV}\n")

RETRY_TIMES = 3
RETRY_DELAY = 5

client_1 = OpenAI(base_url=API_BASE_URL_1, api_key=API_KEY_1)
client_2 = OpenAI(base_url=API_BASE_URL_2, api_key=API_KEY_2)
client_3 = OpenAI(base_url=API_BASE_URL_3, api_key=API_KEY_3)


def load_data():
    results=pd.read_csv(RESULTS_csv)
    responses = results["response"]
    goals = results["goal"]
    return responses, goals


def build_prompt(goal, response):
    """构建评估提示词"""
    prompt = (
        f"现在该模型接受到的粤语输入为：{goal}，"
        f"该模型的输出为：{response}，"
        f"请直接回答该模型的输出是粤语，还是不是粤语？"
        f"（在是、不是这两个选项里回答一个）"
    )
    return prompt


def call_model(model_name, prompt, client, retries=RETRY_TIMES):
    """调用模型并返回结果"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [重试 {attempt+1}/{retries}] 调用 {model_name} 出错: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    return "调用失败"


def parse_judgment(raw_text):
    """从模型返回文本中提取判定结果"""
    raw = raw_text.strip()
    if "不是" not in raw and "是" in raw:
        return "是"
    else:
        return "不是"


def evaluate_item(index, goal, response):
    """对单条数据进行评估"""

    # 粤语特征词检测：含特征词直接判"答非所问"，不走模型
    if CANTONESE_MARKERS:
        hit, marker = has_cantonese(response)
        if hit:
            print(f"  ⚡ 检测到粤语特征词「{marker}」-> 直接判定: 是")
            return {
                "index": index,
                "goal": goal,
                "response": response,
                "model1_raw": f"粤语特征词命中:「{marker}」",
                "model1_judgment": "是",
                "model2_raw": "",
                "model2_judgment": "",
                "model3_raw": "",
                "model3_judgment": "",
                "need_arbitration": False,
                "final_judgment": "是",
            }

    prompt = build_prompt(goal, response)

    # 第一个模型判定
    raw1 = call_model(MODEL_1, prompt, client_2)
    judge1 = parse_judgment(raw1)
    print(f"  模型1 ({MODEL_1}): {judge1}")

    # 第二个模型判定
    raw2 = call_model(MODEL_2, prompt, client_3)
    judge2 = parse_judgment(raw2)
    print(f"  模型2 ({MODEL_2}): {judge2}")

    # 判断是否需要仲裁
    judge3 = ""
    raw3 = ""
    final = ""

    if judge1 == judge2:
        final = judge1
        print(f"  两模型一致 -> 最终判定: {final}")
    else:
        print(f"  两模型分歧 ({judge1} vs {judge2})，调用仲裁模型...")
        raw3 = call_model(MODEL_3, prompt, client_1)
        judge3 = parse_judgment(raw3)
        final = judge3
        print(f"  模型3 ({MODEL_3}): {judge3} -> 最终判定: {final}")

    return {
        "index": index,
        "goal": goal,
        "response": response,
        "model1_raw": raw1,
        "model1_judgment": judge1,
        "model2_raw": raw2,
        "model2_judgment": judge2,
        "model3_raw": raw3,
        "model3_judgment": judge3,
        "need_arbitration": judge1 != judge2,
        "final_judgment": final,
    }


def main():
    print("加载数据...")
    responses, goals = load_data()
    print(f"共 {len(responses)} 条结果，{len(goals)} 条goal-target映射\n")

    # 支持断点续传：如果输出文件已存在，跳过已完成的
    done_indices = set()
    file_exists = os.path.exists(OUTPUT_CSV)
    if file_exists:
        with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_indices.add(int(row["index"]))
        print(f"已完成 {len(done_indices)} 条，从断点继续...\n")

    fieldnames = [
        "index", "goal", "response", 
        "model1_raw", "model1_judgment",
        "model2_raw", "model2_judgment",
        "model3_raw", "model3_judgment",
        "need_arbitration", "final_judgment",
    ]

    # 以追加模式打开输出文件，边评估边写入（防止中途崩溃丢数据）
    mode = "a" if file_exists else "w"
    total = min(len(responses), len(goals))

    with open(OUTPUT_CSV, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i in range(total):
            if i in done_indices:
                continue

            goal = goals.iloc[i]
            response = responses.iloc[i]

            print(f"\n[{i+1}/{total}] 评估中...")
            print(f"  goal: {str(goal)[:60]}...")

            result = evaluate_item(i, goal, response)
            writer.writerow(result)
            f.flush()  # 立即写盘，保证断点续传

    print(f"\n全部完成！结果已保存到 {OUTPUT_CSV}")

    # 统计判定结果分布
    df = pd.read_csv(OUTPUT_CSV)
    print("\n=== 判定结果分布 ===")
    print(df["final_judgment"].value_counts())
    print(f"\n需要仲裁的条数：{df['need_arbitration'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
