# 最新状态：2026-10-11

本地工作区与此次更新前的 GitHub main 一致。此次补公开状态说明，将数据探查脚本的默认目录改为仓库内 `data/DuReaderQG/`，保存输出中的个人目录脱敏。训练、BLEU 数字和原始图表未改变。

已完成 Mengzi-T5 在 DuReaderQG 上的问题生成微调、三档对比与检索 baseline。全量训练 14520 条，保存的 BLEU-4 为 0.458；具体分词、评测集合与计算方法以 Notebook 和 `baseline/bleu_eval.py` 为准。该值不是准确率。

入口：[项目说明](README.md)、[训练与评测 Notebook](notebooks/)、[BLEU 计算](baseline/bleu_eval.py)、[结果图](figures/)。本次未重新训练或生成预测。
