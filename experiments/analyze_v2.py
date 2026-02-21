"""Quick local analysis of v2 results."""
import json, numpy as np, os, glob, sys
from scipy.stats import spearmanr

gen_dir = sys.argv[1] if len(sys.argv) > 1 else "results_v2/generations"

def load_by_method(pattern):
    files = sorted(glob.glob(os.path.join(gen_dir, pattern)))
    composites, densities, words = [], [], []
    for f in files:
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                m = rec["slop_metrics"]
                composites.append(m["composite_slop"])
                densities.append(m["slop_density"])
                words.append(m["word_count"])
    return np.array(composites), np.array(densities), np.array(words)

def bootstrap_ci(data, n=10000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n)]
    boots = np.sort(boots)
    return np.mean(data), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def cohens_d_paired(x, y):
    diff = x - y
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd > 0 else 0.0

def perm_test(x, y, n=10000, seed=42):
    rng = np.random.RandomState(seed)
    diff = x - y
    obs = abs(np.mean(diff))
    count = sum(1 for _ in range(n) if abs(np.mean(diff * rng.choice([-1, 1], size=len(diff)))) >= obs)
    return count / n

# Load all methods
methods = {}
for name, pattern in [
    ("baseline", "baseline_*.jsonl"),
    ("prompt_eng", "prompt_engineering_*.jsonl"),
    ("temp_0.3", "temperature_0.3_*.jsonl"),
    ("temp_0.5", "temperature_0.5_*.jsonl"),
    ("temp_0.7", "temperature_0.7_*.jsonl"),
    ("temp_1.0", "temperature_1.0_*.jsonl"),
    ("logit_-5", "logit_suppression_-5.0_*.jsonl"),
    ("logit_-10", "logit_suppression_-10.0_*.jsonl"),
    ("logit_-20", "logit_suppression_-20.0_*.jsonl"),
    ("caa_0.5", "caa_repeng_0.5_*.jsonl"),
    ("caa_1.0", "caa_repeng_1.0_*.jsonl"),
    ("random_abl", "random_ablation_*.jsonl"),
]:
    c, d, w = load_by_method(pattern)
    if len(c) > 0:
        methods[name] = {"composite": c, "density": d, "words": w}

for alpha in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.5", "2.0", "3.0"]:
    c, d, w = load_by_method(f"neuron_steering_{alpha}_*.jsonl")
    if len(c) > 0:
        methods[f"neuron_{alpha}"] = {"composite": c, "density": d, "words": w}

baseline_c = methods["baseline"]["composite"]

print("=" * 95)
print("V2 RIGOROUS ANTI-SLOP RESULTS (N=225 per method, 3 seeds x 75 prompts)")
print("=" * 95)
header = f"{'Method':<18} {'Composite [95pct CI]':>32} {'Words':>8} {'Cohen_d':>9} {'p-value':>9} {'Sig':>5}"
print(header)
print("-" * 95)

for name in sorted(methods.keys()):
    m = methods[name]
    c = m["composite"]
    mean_c, lo, hi = bootstrap_ci(c)
    mw = np.mean(m["words"])

    if name == "baseline":
        print(f"{name:<18} {mean_c:8.2f} [{lo:6.2f}, {hi:6.2f}] {mw:8.0f}      --        --    --")
    else:
        min_len = min(len(baseline_c), len(c))
        d = cohens_d_paired(baseline_c[:min_len], c[:min_len])
        p = perm_test(baseline_c[:min_len], c[:min_len])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{name:<18} {mean_c:8.2f} [{lo:6.2f}, {hi:6.2f}] {mw:8.0f}  {d:+8.3f}  {p:8.4f}  {sig:>5}")

# Dose-response
print("\n=== NEURON STEERING DOSE-RESPONSE ===")
ns_alphas, ns_means = [], []
for alpha in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]:
    key = f"neuron_{alpha}"
    if key in methods:
        ns_alphas.append(float(alpha))
        ns_means.append(np.mean(methods[key]["composite"]))
if len(ns_alphas) >= 3:
    rho, p_rho = spearmanr(ns_alphas, ns_means)
    print(f"Spearman rho = {rho:.4f}, p = {p_rho:.4f}")
    print(f"Monotonic? {'YES' if p_rho < 0.05 else 'NO'}")

# Circuit stability
print("\n=== CIRCUIT STABILITY ===")
circuit_dir = os.path.join(os.path.dirname(gen_dir), "circuits")
circuits = {}
for f in sorted(glob.glob(os.path.join(circuit_dir, "seed_*_circuit.json"))):
    with open(f) as fh:
        cd = json.load(fh)
    circuits[cd["seed"]] = set(cd["neurons"].keys())
seeds = sorted(circuits.keys())
for i in range(len(seeds)):
    for j in range(i+1, len(seeds)):
        s1, s2 = seeds[i], seeds[j]
        inter = len(circuits[s1] & circuits[s2])
        union = len(circuits[s1] | circuits[s2])
        print(f"  Seed {s1} vs {s2}: Jaccard={inter/max(union,1):.4f} ({inter}/{union})")

print("\n=== KEY FINDINGS ===")
baseline_mean = np.mean(baseline_c)
for name in ["prompt_eng", "neuron_0.0", "neuron_0.4", "logit_-20", "caa_0.5", "random_abl"]:
    if name in methods:
        m = np.mean(methods[name]["composite"])
        pct = (baseline_mean - m) / baseline_mean * 100
        print(f"  {name:20s}: composite={m:.2f} ({pct:+.1f}% vs baseline)")
