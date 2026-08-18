# CMQ Quick-Check Report

**Date:** 2026-08-12 22:35

**Model:** HuggingFaceTB/SmolLM-135M


## 1. Activation Geometry

| Metric | Value | Interpretation |
|---|---|---|
| anisotropy_global | 0.7466 | ✅ moderate |
| anisotropy_pairwise | 0.5557 | strong |
| PCA d90 | 9 / 576 | ratio=0.0156 |
| top10% variance | 0.6024 | |

✅ **Geometry CONFIRMED** — strong anisotropy + low effective dimension

## 2. Weight Low-Rank Structure

| Metric | Value | Interpretation |
|---|---|---|
| mean_energy_top16 | 0.1896 | |
| mean_energy_top32 | 0.3057 | |
| mean_energy_top64 | 0.4748 | ⚠️ moderate |
| mean_energy_top128 | 0.6884 | |
| rank90 stats | mean=256.8, median=244 | |

## 3. Compression Results

| Method | Config | PPL | Δ% | ratio | compression | vs scalar |
|---|---|---:|---:|---:|---:|---|
| **baseline** | FP32/BF16 | **1.11** | — | 1.00 | 1× | — |
| scalar | 8-bit | 1.10 | -1.0% | 0.9899 | 2.0× | — |
| scalar | 4-bit | 1.36 | +22.6% | 1.2264 | 4.0× | — |
| scalar | 3-bit | 8390837.47 | +754911053.5% | 7549111.5348 | 5.3× | — |
| scalar | 2-bit | 188282989.50 | +16939540115.5% | 169395402.1553 | 7.9× | — |
| low-rank | rank 8 | 4467554851.96 | +401939257836.2% | 4019392579.3615 | 37.9× | — |
| low-rank | rank 16 | 458659.18 | +41264783.6% | 412648.8362 | 18.9× | — |
| low-rank | rank 32 | 2070472.20 | +186277200.7% | 1862773.0069 | 9.5× | — |
| low-rank | rank 64 | 1128976594.27 | +101572343064.4% | 1015723431.6437 | 4.7× | — |
| low-rank | rank 128 | 28643980.81 | +2577056202.9% | 25770563.0293 | 2.4× | — |
| **LR+Q** | r16 q4b | 17719851.26 | +1594228533.3% | 15942286.3327 | 67.1× | ✗ (+1299966979.4%) |
| **LR+Q** | r32 q4b | 3139176.33 | +282426919.9% | 2824270.199 | 35.5× | ✗ (+230296747.3%) |
| **LR+Q** | r64 q4b | 54790070.95 | +4929381002.2% | 49293811.0217 | 18.3× | ✗ (+4019519447.4%) |
| **LR+Q** | r32 q3b | 89962908.09 | +8093828787.6% | 80938288.8762 | 46.4× | ✗ (+972.2%) |
| **LR+Q** | r64 q3b | 164395494.69 | +14790417776.3% | 147904178.7629 | 24.1× | ✗ (+1859.2%) |
| **LR+Q** | r16 q2b | 1659488537.71 | +149301712694.5% | 1493017127.9452 | 120.9× | ✗ (+781.4%) |
| **LR+Q** | r32 q2b | 6415497483.89 | +577192755985.7% | 5771927560.8572 | 67.0× | ✗ (+3307.4%) |
| **LR+Q** | r64 q2b | 333469831.46 | +30001784107.2% | 300017842.0721 | 35.4× | ✗ (+77.1%) |

## 4. Conclusions

**Best LR+Q:** `lowrank32_q4` — PPL=3139176.33, compression=35.5×
⚠️ PPL ratio = 2824270.20× (significant degradation)


### Overall Verdict

🟡 **Geometry confirmed, but quality degrades too much.** Try QAT.