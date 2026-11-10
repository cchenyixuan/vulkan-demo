"""Which coarse time is closest to the fine grid's end state? Re 3200 1000^2 (cold) vs 2000^2 (warm from the coarse end)."""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, '.')
from experiment.validation import cavity_reference as reference
ROOT = pathlib.Path('logs/validation/cavity_re3200_v7')
FINE = ROOT/'n2000_k1_float32_xi0p001_eps0p0025_adami_re3200_warm'
COARSE = ROOT/'n1000_k1_float32_xi0p001_eps0p0025_adami_re3200'
def load(run):
    t, f, ke = [], [], []
    for line in open(run/'samples.jsonl', encoding='utf-8'):
        row = json.loads(line); p = run/'samples'/f"s{row['step']:010d}.npz"
        if not p.exists(): continue
        d = np.load(p); t.append(row['time']); ke.append(row['kinetic_energy'])
        f.append(np.concatenate([d['u_marchi_wall_mls'], d['v_marchi_wall_mls']]))
    return np.array(t), np.array(f), np.array(ke)
tf, ff, kf = load(FINE); tc, fc, kc = load(COARSE)
# kinetic energy scale: per particle count differs, normalise by the fine/coarse sample sums
print('KE last fine', kf[-1], 'KE last coarse', kc[-1])
means = np.array([ff[(tf > a) & (tf <= a + 10)].mean(0) for a in np.arange(0, 80, 10.)])
ke_means = np.array([kf[(tf > a) & (tf <= a + 10)].mean() for a in np.arange(0, 80, 10.)])
# tail ratio per 10 t from the last four differences, field-wise Aitken on the last three windows
d1, d2 = means[-2] - means[-3], means[-1] - means[-2]
ratio = float((d1 * d2).sum() / (d1 * d1).sum())
limit = means[-1] + d2 * ratio / (1 - ratio)
ke_ratio = (ke_means[-1] - ke_means[-2]) / (ke_means[-2] - ke_means[-3])
ke_limit = ke_means[-1] + (ke_means[-1] - ke_means[-2]) * ke_ratio / (1 - ke_ratio)
tau = -10 / np.log(ratio)
print(f'tail ratio per 10 t {ratio:.3f} (tau {tau:.1f}), KE ratio {ke_ratio:.3f}, KE limit {ke_limit:.3f}, last window KE {ke_means[-1]:.3f}')
last = means[-1]
marchi = reference.marchi2021(reynolds=3200); ref = np.concatenate([marchi['u'][1], marchi['v'][1]])
rel = lambda a, b: 100*np.sqrt(((a-b)**2).sum()/(b**2).sum())
print(f'fine last window vs Marchi {rel(last, ref):.3f} %, Aitken limit vs Marchi {rel(limit, ref):.3f} %, last-to-limit {rel(last, limit):.3f} %')
print(f"{'coarse t':>10} {'KE':>8} {'to fine last %':>15} {'to fine limit %':>16} {'to Marchi %':>12}")
for a in np.arange(60, 225, 10.):
    m = (tc > a) & (tc <= a + 10)
    c = fc[m].mean(0)
    print(f'{a:4.0f}-{a+10:4.0f} {kc[m].mean():9.3f} {rel(c,last):15.3f} {rel(c,limit):16.3f} {rel(c,ref):12.3f}')
print('fine windows vs limit:', ' '.join(f'{rel(x,limit):.3f}' for x in means))
