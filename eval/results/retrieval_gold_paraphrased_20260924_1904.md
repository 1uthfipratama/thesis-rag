# Retrieval evaluation — 2026-09-24 19:04

Index: 647 chunks, parser `mineru`, embedder `BAAI/bge-small-en-v1.5`, query instruction on. Collections: ['core']. top_k=6, BM25 k=30, dense k=30, RRF k=60.

Question set: `gold_paraphrased.jsonl`. Questions: 20 answerable (17 single-paper, 3 multi-paper).

## All questions

| config | hit@1 | hit@3 | hit@6 | hit@10 | MRR | evidence@6 |
|---|---|---|---|---|---|---|
| bm25 | 0.450 | 0.750 | 0.850 | 1.000 | 0.643 | 0.550 |
| dense | 0.750 | 0.850 | 0.900 | 0.950 | 0.820 | 0.900 |
| hybrid | 0.750 | 0.950 | 1.000 | 1.000 | 0.843 | 0.700 |
| hybrid+cap | 0.750 | 0.950 | 1.000 | 1.000 | 0.843 | 0.650 |
| hybrid+cap2 | 0.750 | 0.950 | 1.000 | 1.000 | 0.843 | 0.650 |
| hybrid+adaptive | 0.750 | 0.950 | 1.000 | 1.000 | 0.843 | 0.700 |

## Targets (PLAN.md Phase 6)

| config | single-paper hit@6 (target ≥ 0.90) | aggregation cov@6 (target ≥ 0.70) | all multi-paper cov@6 | single-paper depth@6 |
|---|---|---|---|---|
| bm25 | 0.824 | 0.389 | 0.389 | 0.412 |
| dense | 0.882 | 0.322 | 0.322 | 0.647 |
| hybrid | 1.000 | 0.389 | 0.389 | 0.598 |
| hybrid+cap | 1.000 | 0.489 | 0.489 | 0.441 |
| hybrid+cap2 | 1.000 | 0.600 | 0.600 | 0.363 |
| hybrid+adaptive | 1.000 | 0.600 | 0.600 | 0.598 |

depth@6 = share of the top-6 chunks that come from the expected paper (what the LLM gets to read).

## hit@6 by question type

| type | n | bm25 | dense | hybrid | hybrid+cap | hybrid+cap2 | hybrid+adaptive |
|---|---|---|---|---|---|---|---|
| aggregation | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| comparison | 5 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| discrepancy | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| equation | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| factoid | 3 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| method | 3 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| numeric | 3 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Misses and partial coverage (hybrid+adaptive)

| id | type | expected | top-6 papers (rank order) | hit@6 | cov@6 | evidence@6 |
|---|---|---|---|---|---|---|
| q01p | numeric | p13 | p13, p12, p08 | 1 | 1.00 | 0 |
| q08p | factoid | p03 | p03, p08 | 1 | 1.00 | 0 |
| q14p | factoid | p05 | p05, p01, p08 | 1 | 1.00 | 0 |
| q33p | comparison | p17 | p17, p08 | 1 | 1.00 | 0 |
| q36p | comparison | p01 | p01, p08, p04, p06 | 1 | 1.00 | 0 |
| q41p | comparison | p16 | p16, p01, p08, p03, p15 | 1 | 1.00 | 0 |
| q46p | aggregation | p01, p02, p13, p03, p05 | p06, p01, p03 | 1 | 0.40 | 1 |
| q50p | aggregation | p02, p13, p03, p05, p04, p15, p12, p17, p06, p01 | p04, p06, p15, p03 | 1 | 0.40 | 1 |

Per-question detail for every config: the `.json` file next to this report.
