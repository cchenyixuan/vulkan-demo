# v7_single acceptance summary

generated 2026-10-09 17:25:32

## 1. CPU command streams (v7 single mode vs v7_single)

all identical: **True** (rc 0)

| case | setting | step | bootstrap_init | bootstrap_compute | defrag | restart |
|---|---|---|---|---|---|---|
| cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml | default | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml | unfused | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml | transfer_copy | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml | default | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml | unfused | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml | transfer_copy | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_gen/case.yaml | default | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_gen/case.yaml | unfused | identical | identical | identical | identical (+allowed fill) | identical |
| cases/lid_driven_cavity_2d_gen/case.yaml | transfer_copy | identical | identical | identical | identical (+allowed fill) | identical |
| cases/cavity3d_8m_b4/case.yaml | default | identical | identical | identical | identical (+allowed fill) | identical |
| cases/cavity3d_8m_b4/case.yaml | unfused | identical | identical | identical | identical (+allowed fill) | identical |
| cases/cavity3d_8m_b4/case.yaml | transfer_copy | identical | identical | identical | identical (+allowed fill) | identical |
| cases/cavity3d_1m/case.yaml | default | identical | identical | identical | identical (+allowed fill) | identical |
| cases/cavity3d_1m/case.yaml | unfused | identical | identical | identical | identical (+allowed fill) | identical |
| cases/cavity3d_1m/case.yaml | transfer_copy | identical | identical | identical | identical (+allowed fill) | identical |

## 2-3. GPU bitwise (canonical lists) and invariants

| battery | setting | S vs V | S vs C | drift S/V/C | overflow S/V/C | fps S/V/C |
|---|---|---|---|---|---|---|
| n250_200 | default | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 4503/3413/4311 |
| n250_200 | unfused | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 3132/2524/3018 |
| n250_999 | default | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 4419/3226/4218 |
| n250_999 | unfused | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 3064/2506/2951 |
| cavity2d_1m_200 | default | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 753/707/742 |
| cavity2d_1m_200 | unfused | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 434/418/431 |
| cavity3d_8m_b4_200 | default | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 15/15/15 |
| cavity3d_8m_b4_200 | unfused | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 7/7/7 |
| cavity3d_1m_200 | default | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 103/101/102 |
| cavity3d_1m_200 | unfused | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 50/49/49 |
| n250_adami_200 | default | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 3295/2689/3185 |
| n250_adami_200 | unfused | BIT-IDENTICAL (no-mask BIT-IDENTICAL) | BIT-IDENTICAL | 0/0/0 | 0/0/0 | 2453/2120/2388 |

## 4. Validation layer (default settings, 2000 steps, v7_single)

| case | layer loaded | messages | drift | overflow | fps |
|---|---|---|---|---|---|
| cavity2d_1m | True | {'ERROR': 0, 'WARNING': 0, 'INFO': 0, 'VERBOSE': 0} (stderr lines 0) | 0 | {} | 710.9165 |
| cavity3d_8m_b4 | True | {'ERROR': 0, 'WARNING': 0, 'INFO': 0, 'VERBOSE': 0} (stderr lines 0) | 0 | {} | 14.7398 |

