"""How much accuracy does each 10 t of a warm-started fine run buy? Measured on the finished Re 3200 runs."""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, '.')
from experiment.validation import cavity_reference as reference

ROOT = pathlib.Path('logs/validation')
def load(run):
    times, u, v, ke = [], [], [], []
    for line in open(run / 'samples.jsonl', encoding='utf-8'):
        row = json.loads(line)
        path = run / 'samples' / f"s{row['step']:010d}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        times.append(row['time']); ke.append(row['kinetic_energy'])
        u.append(data['u_marchi_wall_mls']); v.append(data['v_marchi_wall_mls'])
    return np.array(times), np.array(u), np.array(v), np.array(ke)

def rel(a, b):
    return 100 * np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum())

def report(label, run, re_value, width):
    times, u, v, ke = load(run)
    print(f"\n{label}: {len(times)} samples, t {times[0]:.2f}..{times[-1]:.2f}")
    table = reference.marchi2021(reynolds=re_value)

    u_ref, v_ref = table['u'][1], table['v'][1]
    last = times >= times[-1] - width
    u_last, v_last = u[last].mean(0), v[last].mean(0)
    print(f"{'window':>16} {'L2u %':>7} {'L2v %':>7} {'to-last u %':>11} {'to-last v %':>11} {'KE':>9}")
    edges = np.arange(times[0] - 0.2, times[-1] + 0.1, width)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (times > lo) & (times <= hi + 1e-9)
        if m.sum() < 3: continue
        um, vm = u[m].mean(0), v[m].mean(0)
        print(f"{lo:7.1f}-{hi:7.1f} {rel(um,u_ref):7.3f} {rel(vm,v_ref):7.3f} {rel(um,u_last):11.3f} {rel(vm,v_last):11.3f} {ke[m].mean():9.4f}")

report('Re 3200 2000^2 warm', ROOT/'cavity_re3200_v7/n2000_k1_float32_xi0p001_eps0p0025_adami_re3200_warm', 3200, 10.0)
report('Re 3200 1000^2', ROOT/'cavity_re3200_v7/n1000_k1_float32_xi0p001_eps0p0025_adami_re3200', 3200, 20.0)
