# Data-efficiency curve

How execution accuracy varies with the amount of fine-tuning data, scored on the same 500 held-out WikiSQL test examples with identical eval code at every point.

| Train examples | Execution accuracy | Validity | Exact match | Gain | Share of total gain | Train time |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **39.8%** | 55.2% | 3.4% | +0.0 pts | 0% | — |
| 200 | **77.2%** | 98.0% | 68.4% | +37.4 pts | 82% | 12 min |
| 1,000 | **81.8%** | 99.4% | 74.0% | +42.0 pts | 92% | 50 min |
| 6,000 | **85.4%** | 99.8% | 77.8% | +45.6 pts | 100% | 112 min |

> Epochs held constant while training-set size varies, so optimiser steps scale with data. Data quantity and training duration are confounded by design; this mirrors the practical decision.
