# The viscous operator on the exact lattice (experiment/v8/_viscous_operator_lattice.py)

v = (sin(2 pi y), 0) on the periodic 200² lattice, calibrated volume, KCG with xi = 0.001; the table gives a_x / (nu lap v) = the operator's nu_eff / nu (mean over the particles where |lap v| > 0.3 k²; min–max in brackets).

| h/dx | V / dx² | L diagonal | pairs / particle | ε² = 0 h² | ε² = 0.0025 h² | ε² = 0.01 h² |
|---|---|---|---|---|---|---|
| 5 (200²) | 1.1293 | 0.8850 | 80.0 | 0.9737 [0.9737, 0.9737] | 0.9577 [0.9577, 0.9577] | 0.9137 [0.9137, 0.9137] |
| 4 (200²) | 1.2175 | 0.8218 | 48.0 | 0.9392 [0.9392, 0.9392] | 0.9259 [0.9259, 0.9259] | 0.8886 [0.8886, 0.8886] |
| 3.5 (203²) | 1.3035 | 0.7680 | 36.0 | 0.8985 [0.8985, 0.8985] | 0.8872 [0.8872, 0.8872] | 0.8552 [0.8552, 0.8552] |
