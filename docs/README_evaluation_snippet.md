## 检索策略评估

项目提供增强版评估脚本 `scripts/evaluate_with_topk.py`，用于对不同检索策略进行离线评估。

当前支持三组对比：

- BM25
- Hybrid Search：BM25 + Vector Search + RRF
- Hybrid Search + Reranker

评估指标包括：

- Top-1 Hit Rate
- Top-3 Hit Rate
- Top-5 Hit Rate
- MRR
- Faithfulness
- Answer Relevancy
- 平均检索耗时
- 平均生成耗时

### 构建评估集

```bash
python scripts/build_eval_dataset_with_source.py \
  --knowledge_id 1 \
  --num_questions 100 \
  --output scripts/eval_dataset.json
```

生成后需要人工检查 `question`、`ground_truth` 与 `source_content` 是否一致。

### 运行评估

完整评估：

```bash
python scripts/evaluate_with_topk.py \
  --knowledge_id 1 \
  --dataset scripts/eval_dataset.json \
  --output scripts/eval_result_topk.json
```

快速调试，仅跑前 5 条：

```bash
python scripts/evaluate_with_topk.py \
  --knowledge_id 1 \
  --dataset scripts/eval_dataset.json \
  --output scripts/eval_result_topk.json \
  --limit 5
```
