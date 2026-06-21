import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取并修复列类型
df = pd.read_csv("eval_set.csv")
df["prediction"] = df["prediction"].astype(object)


def extract_candidate_spans(text, window_size=30):
    """
    用滑动窗口从 context 中提取候选片段
    window_size=30 表示每个候选片段约30个字符（中文约15-20个字）
    """
    if not isinstance(text, str):
        return []
    text = text.strip()
    if len(text) <= window_size:
        return [text]
    candidates = []
    step = window_size // 2  # 步长为窗口的一半，保证重叠覆盖
    for i in range(0, len(text) - window_size + 1, step):
        candidates.append(text[i:i + window_size])
    # 确保最后一段也被覆盖
    if len(text) - window_size < len(text) and (len(text) - window_size) % step != 0:
        candidates.append(text[-window_size:])
    return candidates


def predict_answer(question, context, window_size=40):
    if pd.isna(context) or not context:
        return ""

    # 如果 context 很短，直接返回整个 context
    if len(context) <= 50:
        return context.strip()

    candidates = extract_candidate_spans(context, window_size=window_size)
    if not candidates:
        return context[:50]

    # 限制候选数量，避免性能问题
    if len(candidates) > 50:
        candidates = candidates[:50]

    vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None)
    try:
        tfidf_matrix = vectorizer.fit_transform([question] + candidates)
        q_vec = tfidf_matrix[0]
        c_vecs = tfidf_matrix[1:]
        similarities = cosine_similarity(q_vec, c_vecs).flatten()
        best_idx = similarities.argmax()
        return candidates[best_idx].strip()
    except Exception as e:
        # 保底方案：返回 context 前50个字
        return context[:50]


print("开始生成 Baseline 预测（共100条）...")
for index, row in df.iterrows():
    question = row["question"]
    context = row["context"]
    prediction = predict_answer(question, context)
    df.at[index, "prediction"] = prediction
    if (index + 1) % 10 == 0:
        print(f"已处理 {index + 1}/100 条")

df.to_csv("eval_set.csv", index=False, encoding="utf-8-sig")
print("✅ 预测完成！结果已保存至 eval_set.csv")
print("\n前10条预测示例：")
print(df[["question", "prediction"]].head(10))