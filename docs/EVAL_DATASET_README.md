# 评估集说明：中华人民共和国知识产权海关保护条例

本评估集基于《中华人民共和国知识产权海关保护条例》构造，共 30 条，其中 27 条可回答问题、3 条无答案/拒答问题。

## 题型分布

- `exact_clause`: 6 条
- `semantic_rewrite`: 7 条
- `scenario`: 6 条
- `process`: 4 条
- `multi_clause`: 4 条
- `no_answer`: 3 条

## 文件说明

- `eval_dataset_ip_customs_balanced_30.json`：完整 30 条，包含无答案题，适合端到端问答和拒答测试。
- `eval_dataset_ip_customs_answerable_27.json`：去掉无答案题，适合先跑 Top-K / MRR 检索测试。
- `evaluate_retrieval_only.py`：只测检索，不调用 LLM，不跑 RAGAS，速度更快。
- `expected_chunk_ids` 为空的样本表示未强行填充未确认 chunk_id，脚本会回退到 document_id 级命中，适合先做整体检索测试。
- 不建议只使用 `exact_clause`，否则会天然偏向 BM25。
