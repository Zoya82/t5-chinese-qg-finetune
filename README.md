# 中文问题生成：Mengzi-T5 微调与评测（DuReaderQG）

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

基于 **Mengzi-T5-Base** 在百度 **DuReaderQG** 数据集上微调，完成中文**问题生成**任务（给定篇章与答案，生成对应问题），并设置"未微调 / 小样本 / 全量"三档消融与检索式 baseline 对比，用 BLEU-1~4 评估。

## 📊 结果（验证集，BLEU）

| 实验 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| 未微调 | 0.000 | 0.000 | 0.000 | 0.000 |
| 小样本微调 | 0.378 | 0.372 | 0.361 | 0.274 |
| **全量微调** | **0.684** | **0.639** | **0.524** | **0.458** |

全量训练 14520 条样本；loss 曲线、BLEU 对比图、与 baseline 的对比见 [`figures/`](figures)。

## 🔧 方法

数据读取与异常样本过滤 → 训练参数配置 → Mengzi-T5 微调 → 验证集生成预测 → 计算 BLEU-1~4 → 与检索/抽取式 baseline 对比，统一打分口径。

## 🗂 结构

```
notebooks/  03 正式训练与导出 · 04 全量训练与评测
baseline/   baseline_tfidf.py（检索式 baseline）· bleu_eval.py（字符级 BLEU，统一口径）
src/        inspect_data.py（数据探查）
data/       eval_set.csv（评测样本）
figures/    loss 曲线 / BLEU 对比图
```

## ▶️ 运行

```bash
pip install -r requirements.txt
# 下载 DuReaderQG 数据集放到 data/DuReaderQG/（见下），按 notebooks/03 → 04 运行
```

底模 `Mengzi-T5-Base`（[Langboat/mengzi-t5-base](https://huggingface.co/Langboat/mengzi-t5-base)）；Apple MPS / 单卡可跑。

## 📁 数据集

- **DuReaderQG**（百度开源），版权归原作者，未随仓库分发（见 `.gitignore`）。
- 自行下载 `dev.json` / `train.json` 放入 `data/DuReaderQG/` 后即可运行。

## License

[MIT](LICENSE)
