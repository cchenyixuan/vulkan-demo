# Re 5000, 2500²: ψ_min and the primary vortex centre (tab:cavity5000)

Window t 100.701–103.314: 14 sample steps (33,567,000–34,438,000), particle snapshots inside it at steps 33,633,000, 33,966,000, 34,299,000 (t 100.899, 101.898, 102.897), the same in both runs. Method of paper Tables 4 / 5 (adami_k8_report → cavity_analysis.stream_function_minimum): time-averaged MLS grid of u, 513 × 513 over the wall rows, ψ integrated upward from the bottom wall (ψ = 0), interior minimum refined by a 5 × 5 quadratic surface; wall frame.

| | ψ_min | x | y |
|---|---|---|---|
| Marchi 2021, Table 27 | −0.1222 | 0.5151 | 0.5353 |
| 2x5090 (3-snapshot mean) | −0.1225 | 0.5150 | 0.5350 |
| 5090 + XTX (3-snapshot mean) | −0.1225 | 0.5149 | 0.5351 |
| difference (5090 + XTX − 2x5090) | -7.3e-06 | -9.3e-05 | +4.0e-05 |

Per snapshot (each through the same function, one snapshot in the window), min … max:

| | ψ_min | x | y |
|---|---|---|---|
| 2x5090 | −0.1225 … −0.1225 | 0.5149 … 0.5152 | 0.5350 … 0.5351 |
| 5090 + XTX | −0.1225 … −0.1225 | 0.5149 … 0.5150 | 0.5351 … 0.5351 |
| difference at the same snapshot | -1.5e-05 … +9.5e-07 | -1.6e-04 … +2.5e-05 | -1.3e-06 … +6.5e-05 |

Snapshots one by one (ψ_min, x, y):

- 2x5090, t 100.899: -0.122502, 0.51491, 0.53501
- 2x5090, t 101.898: -0.122488, 0.51519, 0.53504
- 2x5090, t 102.897: -0.122479, 0.51502, 0.53506
- 5090 + XTX, t 100.899: -0.122509, 0.51494, 0.53508
- 5090 + XTX, t 101.898: -0.122503, 0.51504, 0.53509
- 5090 + XTX, t 102.897: -0.122478, 0.51486, 0.53506

Section 4 check (analyse_window's dense-line extrema on the same 14 samples vs re5000_mixed/summary.json):

- 5090 + XTX u_min: -0.447973 @ 0.07348 vs -0.447973 @ 0.07348 (max |Δ| 0.0e+00)
- 5090 + XTX v_max: 0.447983 @ 0.07914 vs 0.447983 @ 0.07914 (max |Δ| 0.0e+00)
- 5090 + XTX v_min: -0.577059 @ 0.95791 vs -0.577059 @ 0.95791 (max |Δ| 0.0e+00)
- 2x5090 u_min: -0.447945 @ 0.07349 vs -0.447945 @ 0.07349 (max |Δ| 0.0e+00)
- 2x5090 v_max: 0.447992 @ 0.07915 vs 0.447992 @ 0.07915 (max |Δ| 0.0e+00)
- 2x5090 v_min: -0.577120 @ 0.95788 vs -0.577120 @ 0.95788 (max |Δ| 0.0e+00)
