#!/usr/bin/env python3
"""字符级 BLEU 评测（与 T5 实验完全相同的打分口径）。

用法 1（命令行）：把你的 baseline 预测填进 eval_set.csv 的 prediction 列后运行：
    python bleu_eval.py eval_set.csv
会打印 BLEU-1~4。

用法 2（导入）：
    from bleu_eval import bleu_scores
    scores = bleu_scores(predictions, references)   # 两个等长字符串列表

注意：必须用这份函数给 baseline 打分，才能和 T5 结果放进同一张对比表。
不要换成 nltk / sacrebleu 等词级 BLEU，分数体系不同、不可比。
"""
from __future__ import annotations

import math
import sys

import numpy as np


def char_tokens(text):
    # 中文按“字符”切分，和 T5 实验一致
    return list(str(text).strip())


def modified_precision(pred_tokens, ref_tokens, n):
    if len(pred_tokens) < n:
        return 0.0
    pred_ngrams = {}
    ref_ngrams = {}
    for i in range(len(pred_tokens) - n + 1):
        gram = tuple(pred_tokens[i:i + n])
        pred_ngrams[gram] = pred_ngrams.get(gram, 0) + 1
    for i in range(len(ref_tokens) - n + 1):
        gram = tuple(ref_tokens[i:i + n])
        ref_ngrams[gram] = ref_ngrams.get(gram, 0) + 1
    overlap = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in pred_ngrams.items())
    total = max(1, sum(pred_ngrams.values()))
    return overlap / total


def sentence_bleu(prediction, reference, max_n=4, smooth=1e-9):
    pred_tokens = char_tokens(prediction)
    ref_tokens = char_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    bp = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    precisions = [max(modified_precision(pred_tokens, ref_tokens, n), smooth) for n in range(1, max_n + 1)]
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def corpus_bleu(predictions, references, max_n=4):
    scores = [sentence_bleu(pred, ref, max_n=max_n) for pred, ref in zip(predictions, references)]
    return float(np.mean(scores)) if scores else 0.0


def bleu_scores(predictions, references):
    """返回 {'bleu_1':..,'bleu_2':..,'bleu_3':..,'bleu_4':..}"""
    return {f"bleu_{n}": corpus_bleu(predictions, references, max_n=n) for n in range(1, 5)}


if __name__ == "__main__":
    import pandas as pd

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "eval_set.csv"
    df = pd.read_csv(csv_path)
    if "prediction" not in df.columns or df["prediction"].fillna("").str.strip().eq("").all():
        print("⚠️  prediction 列为空，请先把你的 baseline 预测填进 eval_set.csv 再运行。")
        sys.exit(1)
    preds = df["prediction"].fillna("").astype(str).tolist()
    refs = df["reference"].fillna("").astype(str).tolist()
    scores = bleu_scores(preds, refs)
    print(f"评测样本数: {len(df)}")
    for n in range(1, 5):
        print(f"BLEU-{n}: {scores[f'bleu_{n}']:.4f}")
