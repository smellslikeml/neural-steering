# V2 Rigorous Anti-Slop Experiment Specification

**Version**: 2.0
**Date**: 2026-02-20
**Status**: Design spec (pre-implementation)
**Goal**: Address all reviewer concerns from v1 rejection and produce publication-quality results.

---

## 1. Reviewer Concerns Addressed

| # | V1 Problem | V2 Fix | Section |
|---|-----------|--------|---------|
| 1 | N=10 test prompts | N=75 diverse held-out prompts across 5 categories | 2 |
| 2 | No dose-response monotonicity test | Monotonicity tests + fine-grained sweeps for ALL methods | 5.1 |
| 3 | Unvalidated composite_slop metric | GPT-4 judge (1-5 naturalness) + Spearman correlation | 7 |
| 4 | Unfair CAA comparison (our implementation) | repeng library (vgel) with standard protocol | 4.4 |
| 5 | Missing baselines | Prompt engineering, temperature, logit suppression, random ablation | 4 |
| 6 | Single seed, single model | 3 seeds (42, 123, 456); Llama-3.1-8B-Instruct primary | 3 |
| 7 | Only 5 perplexity passages | WikiText-2 test set, first 100 passages | 6 |
| 8 | No word-count normalization | slop_density = composite_slop / word_count as primary metric | 5.2 |
| 9 | No statistical tests | Bootstrap CI, permutation tests, Cohen's d, Bonferroni | 8 |

---

## 2. Test Prompt Pool (N=75, held-out)

**Critical**: These prompts MUST NOT overlap with any discovery prompts (SLOP_POSITIVE, SLOP_NEGATIVE, or the 20 TOPICS used to generate them). All prompts are plain instructions with no style guidance, to measure how steering alone changes output.

### 2.1 Creative Writing (15 prompts)
```python
CREATIVE_WRITING = [
    "Write a short story about a lighthouse keeper who discovers something unusual.",
    "Describe a rainy afternoon from the perspective of a cat.",
    "Write about a conversation between two strangers on a delayed flight.",
    "Tell the story of someone returning to their hometown after twenty years.",
    "Write a scene set in a bakery at 4 AM.",
    "Describe the last day of summer from a child's point of view.",
    "Write about a musician performing to an empty room.",
    "Tell a story that begins and ends with the same sentence.",
    "Write about two neighbors who communicate only through notes.",
    "Describe a city during a power outage.",
    "Write about someone who finds an old letter in a used book.",
    "Tell the story of a road trip that goes wrong.",
    "Write about a teacher on their first day at a new school.",
    "Describe a meal that changes someone's mind about something.",
    "Write about the moment just before a thunderstorm.",
]
```

### 2.2 Technical Explanation (15 prompts)
```python
TECHNICAL_EXPLANATION = [
    "Explain how a compiler transforms source code into machine instructions.",
    "Describe how GPS satellites determine your location on Earth.",
    "Explain the mechanics of how an airplane wing generates lift.",
    "Describe how vaccines train the immune system to fight disease.",
    "Explain how fiber optic cables transmit data using light.",
    "Describe the process by which rivers form deltas at their mouths.",
    "Explain how noise-canceling headphones work.",
    "Describe how CRISPR gene editing modifies DNA sequences.",
    "Explain the physics behind why ice is slippery.",
    "Describe how a neural network learns to classify images.",
    "Explain how tides are caused by gravitational forces.",
    "Describe the chemical process of how batteries store and release energy.",
    "Explain how MRI machines create images of the human body.",
    "Describe how earthquakes propagate through different types of rock.",
    "Explain how refrigeration works at a molecular level.",
]
```

### 2.3 Advice and Recommendations (15 prompts)
```python
ADVICE_GIVING = [
    "What should someone consider before adopting a rescue dog?",
    "How should a person prepare for a job interview in a field they're switching to?",
    "What are practical ways to reduce household food waste?",
    "How should someone approach learning to cook if they've never done it?",
    "What should a first-time homebuyer know before making an offer?",
    "How can someone improve their public speaking skills?",
    "What should you consider before starting a small business?",
    "How should someone approach difficult conversations with family members?",
    "What are effective strategies for managing personal finances on a tight budget?",
    "How should a beginner approach learning a musical instrument?",
    "What should someone think about before moving to a new country?",
    "How can a person build better daily habits?",
    "What should you know before investing in the stock market for the first time?",
    "How should someone approach writing a resume after a career gap?",
    "What are practical ways to reduce stress during busy work periods?",
]
```

### 2.4 Summarization (15 prompts)
```python
SUMMARIZATION = [
    "Summarize the main arguments for and against nuclear power as an energy source.",
    "Describe the key events of the Apollo 11 moon landing mission.",
    "Summarize how the internet evolved from ARPANET to the modern web.",
    "Describe the major causes and consequences of the 2008 financial crisis.",
    "Summarize the current scientific understanding of how memory works in the brain.",
    "Describe the key principles behind how electric vehicles differ from combustion engines.",
    "Summarize the history of antibiotics from penicillin to modern resistance concerns.",
    "Describe the main theories about what caused the extinction of the dinosaurs.",
    "Summarize the debate around standardized testing in education.",
    "Describe how the global supply chain works for a typical consumer product.",
    "Summarize the scientific consensus on the health effects of processed foods.",
    "Describe the major milestones in the history of space exploration.",
    "Summarize how different countries approach universal healthcare.",
    "Describe the key factors that influence weather patterns and forecasting.",
    "Summarize the environmental impact of fast fashion.",
]
```

### 2.5 Open-Ended Questions (15 prompts)
```python
OPEN_ENDED = [
    "What makes a city feel alive?",
    "Why do people collect things?",
    "What changes when you learn a second language?",
    "Why do some friendships last decades while others fade quickly?",
    "What makes a good apology?",
    "Why is it hard to change your mind about something you believe strongly?",
    "What role does boredom play in creativity?",
    "Why do people feel nostalgic for times they never experienced?",
    "What makes a house feel like a home?",
    "Why do some books stay with you long after you read them?",
    "What changes about a person when they become a parent?",
    "Why is it easier to give advice than to follow it?",
    "What makes a conversation memorable?",
    "Why do people root for underdogs?",
    "What does it mean to age well?",
]
```

### 2.6 Validation of non-overlap
Before running: programmatically verify zero lexical overlap between test prompts and the 20 TOPICS used in discovery:
```python
DISCOVERY_TOPICS = {
    "education", "technology", "teamwork", "renewable energy", "leadership",
    "globalization", "emotional intelligence", "social media", "creativity",
    "diversity in the workplace", "artificial intelligence", "mental health",
    "climate change", "remote work", "healthcare innovation",
    "financial literacy", "urban planning", "scientific research",
    "automation", "digital privacy",
}
# Assert: for each test prompt, no DISCOVERY_TOPIC appears as a substring
```

---

## 3. Multi-Seed Protocol

### Seeds
- **Discovery seeds**: 42, 123, 456
- For EACH seed: re-run contrastive neuron discovery from scratch
- This produces 3 independent circuits per seed

### Reporting
- All metrics reported as: mean +/- std across 3 seeds
- Per-seed results stored separately for transparency
- Circuit stability analysis: Jaccard similarity of discovered neurons across seeds

### Generation
- Use `do_sample=False` (greedy decoding) for reproducibility
- Temperature parameter only varied explicitly in the temperature baseline (Section 4.2)

---

## 4. Methods Under Comparison

### 4.1 Unsteered Baseline
- Model generates with no intervention
- This is the reference point for all comparisons

### 4.2 Prompt Engineering
System-prompt-based anti-slop instruction, prepended to each test prompt:
```python
ANTISLOP_SYSTEM_PROMPT = (
    "Write in a direct, natural style. Avoid cliches and filler phrases. "
    "Do not use bullet points or numbered lists unless specifically asked. "
    "Avoid words like: delve, tapestry, testament, paradigm, synergy, "
    "multifaceted, holistic, nuanced, realm, landscape, myriad, plethora, "
    "utilize, leverage, facilitate, encompass. "
    "Do not use em dashes. Vary your sentence length. "
    "Do not start with sycophantic phrases like 'Great question'."
)
```
No sweep needed -- this is a single configuration.

### 4.3 Temperature Reduction
Sweep temperatures: [0.3, 0.5, 0.7, 1.0]
- Use `do_sample=True, temperature=T, top_p=0.95` for T < 1.0
- Baseline (1.0) uses greedy for consistency with other methods
- Note: temperature affects randomness, not style. This tests reviewer's hypothesis that simple temperature reduction could achieve similar slop reduction.

### 4.4 Logit Suppression
Directly bias logits for known slop tokens during generation:
```python
LOGIT_BIAS_STRENGTHS = [-5, -10, -20]  # applied to slop token logits

# Tokens to suppress: tokenize each word in TIER1_SLOP + TIER2_SLOP
# For each word, find ALL token IDs that encode it (including with/without
# leading space, capitalized variants)
# Apply bias to those token IDs at every generation step via LogitsProcessor
```
This is the strongest possible "just ban the words" baseline. If neuron steering doesn't beat this, the method has no advantage over simple logit manipulation.

### 4.5 RepEng CAA (Contrastive Activation Addition)
Use the `repeng` library by vgel for proper implementation:
```python
# pip install repeng
from repeng import ControlVector, ControlModel

# Train control vector using SAME discovery prompts as neuron steering
# (SLOP_POSITIVE, SLOP_NEGATIVE)
# This ensures a fair comparison: same signal, different method

# Sweep alpha: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
# repeng applies to residual stream across all layers by default
```
**Important**: If repeng cannot be installed or is incompatible with the model, fall back to our manual implementation BUT clearly document this in the paper as a limitation. The reviewer specifically called out our CAA implementation as unfair.

**Fallback CAA implementation**: If repeng is used, also run our manual CAA for comparison to verify they produce similar results. If they differ substantially, report both.

### 4.6 Neuron Steering (Our Method)
```python
# Discover circuit via contrastive neuron discovery
# Sweep multiplier alpha: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# alpha=1.0 = normal (no steering), alpha=0.0 = full ablation
# Also test amplification: [1.5, 2.0, 3.0]
```

### 4.7 Random Neuron Ablation (Negative Control)
```python
# For EACH seed:
#   Sample N random neurons matching the layer distribution of the discovered circuit
#   Ablate with alpha=0.0
#   Run on all 75 test prompts
# This tests: "is the effect specific to discovered neurons or would any ablation work?"
# Run 5 random samples per seed to get variance estimate
```

---

## 5. Metrics

### 5.1 Primary: Slop Density (word-count normalized)

```python
slop_density = composite_slop / word_count
```

This removes the length confound identified by the reviewer. A 500-word essay with composite_slop=10 is less sloppy than a 100-word essay with composite_slop=10.

Also report raw `composite_slop` for backward compatibility, but `slop_density` is the primary comparison metric.

### 5.2 Component Metrics (for interpretability)
Report all sub-components from `count_slop()`:
- `tier1_count`, `tier2_count`, `tier3_count` (raw slop word/phrase counts)
- `em_dash_count` (structural marker)
- `list_ratio` (bullet/numbered list tendency)
- `ttr` (type-token ratio, vocabulary diversity)
- `sentence_cv` (sentence length variation / burstiness)
- `start_diversity` (sentence opening diversity)

### 5.3 Dose-Response Monotonicity Test
For each method with a sweep parameter:
```python
# Compute Spearman rank correlation between sweep parameter and slop_density
# For neuron steering: expect negative correlation (lower alpha = less slop)
# For logit suppression: expect negative correlation (stronger bias = less slop)
# For CAA: expect negative correlation (higher alpha = less slop)
# Report rho and p-value
# A method with non-monotonic dose-response is harder to use reliably
```

---

## 6. Perplexity Benchmark

### 6.1 WikiText-2 Test Set
```python
from datasets import load_dataset

wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Filter: keep only passages with >= 50 tokens (skip empty lines, headers)
# Take the first 100 qualifying passages
# Truncate each to 256 tokens max (to keep compute manageable)

PERPLEXITY_PASSAGES = []
for text in wikitext["text"]:
    text = text.strip()
    if len(text.split()) >= 50:
        PERPLEXITY_PASSAGES.append(text)
    if len(PERPLEXITY_PASSAGES) >= 100:
        break
```

### 6.2 Perplexity Computation
```python
def compute_perplexity(model, tokenizer, texts, max_length=256):
    """Standard sliding-window perplexity on a list of texts."""
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        nll = outputs.loss.item() * (input_ids.shape[1] - 1)
        total_nll += nll
        total_tokens += input_ids.shape[1] - 1
    return math.exp(total_nll / total_tokens)
```

### 6.3 Perplexity Reporting
- Report absolute perplexity AND perplexity ratio (steered / baseline)
- Threshold: ppl_ratio > 1.10 = significant degradation (flag in results)
- Compute perplexity at EVERY sweep point for every method
- This enables Pareto frontier plots (slop_reduction vs ppl_degradation)

---

## 7. Human Evaluation Proxy (GPT-4 Judge)

### 7.1 Design
Use GPT-4 as a calibrated naturalness judge on a stratified subset of outputs.

**Subset**: 20 prompts (4 per category), evaluated across all methods at their "best" operating point (best slop reduction with ppl_ratio < 1.05). This gives ~140 (prompt, method, output) triples.

### 7.2 GPT-4 Judging Protocol
```python
JUDGE_PROMPT = """You are evaluating the naturalness and quality of AI-generated text.

Rate the following text on a scale of 1-5:
1 = Extremely formulaic, full of cliches, feels like generic AI output
2 = Noticeably artificial, uses common AI patterns but occasionally natural
3 = Mixed: some natural passages, some formulaic ones
4 = Mostly natural, reads like competent human writing with minor AI tells
5 = Highly natural, varied, engaging. Hard to distinguish from skilled human writing.

Focus on:
- Vocabulary diversity (does it repeat the same fancy words?)
- Structural variety (does it always use bullet points or numbered lists?)
- Sentence rhythm (are all sentences similar length?)
- Opening and transitions (does it use cliches like "in today's world"?)
- Overall feel (does it read like a human or a template?)

DO NOT penalize for being concise or informal. Natural human writing is often short and direct.

TEXT TO EVALUATE:
{text}

Your rating (1-5) and a one-sentence justification:"""
```

### 7.3 Validation
- Compute Spearman rank correlation between GPT-4 naturalness score (inverted: 6 - score) and composite_slop
- If correlation > 0.5: composite_slop is a reasonable proxy for human judgment
- If correlation < 0.3: composite_slop may not be measuring what we think
- Report the correlation with 95% CI

### 7.4 Inter-Rater Agreement
- Run each judgment 3 times (temperature=0.3) to measure GPT-4 self-consistency
- Report Krippendorff's alpha for inter-rater reliability

---

## 8. Statistical Framework

### 8.1 Bootstrap Confidence Intervals
```python
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))
    boot_stats = np.sort(boot_stats)
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return statistic(data), lower, upper
```

All metrics reported as: **mean [95% CI lower, upper]**

### 8.2 Paired Permutation Tests
For pairwise method comparisons (e.g., neuron steering vs logit suppression):
```python
def paired_permutation_test(x, y, n_permutations=10000, seed=42):
    """Two-sided paired permutation test. H0: no difference in means."""
    rng = np.random.RandomState(seed)
    diff = x - y
    observed = np.mean(diff)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_diff = np.mean(diff * signs)
        if abs(perm_diff) >= abs(observed):
            count += 1
    return count / n_permutations  # p-value
```

### 8.3 Effect Size (Cohen's d)
```python
def cohens_d(x, y):
    """Paired Cohen's d effect size."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)
```

Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large.

### 8.4 Multiple Comparison Correction
- K methods being compared pairwise: K*(K-1)/2 comparisons
- With 6 methods (baseline, prompt eng, temp, logit, CAA, neuron): 15 pairwise comparisons
- Apply **Bonferroni correction**: alpha_adj = 0.05 / 15 = 0.0033
- Also report Holm-Bonferroni (less conservative) as secondary

### 8.5 Power Analysis
With N=75 prompts and 3 seeds (effective N=225 observations per method):
- For paired test, d=0.3 (small effect): power ~ 0.95
- For paired test, d=0.2 (very small): power ~ 0.72
- Adequate for detecting meaningful differences

---

## 9. Experiment Structure

### 9.1 Phase 1: Circuit Discovery (per seed)
For each seed in [42, 123, 456]:
1. Set all random seeds (torch, numpy, python random)
2. Discover anti-slop circuit: `discover_contrastive(SLOP_POSITIVE, SLOP_NEGATIVE, top_k=200)`
3. Save circuit (neuron indices + attributions)
4. Compute circuit stability: Jaccard overlap between seeds

**Output**: `circuits/seed_{seed}_circuit.json`

### 9.2 Phase 2: Generation (per method, per seed)
For each (method, seed, sweep_param):
1. Load circuit for this seed (if method uses circuit)
2. Generate on all 75 test prompts (max_tokens=500, greedy unless temperature method)
3. Save all raw generations

**Output**: `generations/{method}_{seed}_{param}.jsonl`

### 9.3 Phase 3: Measurement
For each generation file:
1. Run `count_slop()` on every generation
2. Compute `slop_density = composite_slop / word_count`
3. Save per-prompt metrics

**Output**: `metrics/{method}_{seed}_{param}_metrics.json`

### 9.4 Phase 4: Perplexity
For each (method, seed, sweep_param):
1. Apply steering/intervention
2. Compute perplexity on 100 WikiText-2 passages
3. Record ppl and ppl_ratio

**Output**: `perplexity/{method}_{seed}_{param}_ppl.json`

### 9.5 Phase 5: GPT-4 Judge
1. Select 20-prompt stratified subset
2. For each method at its best operating point: collect generations
3. Send to GPT-4 for scoring (3 runs each)
4. Compute correlations with composite_slop

**Output**: `gpt4_judge/scores.json`

### 9.6 Phase 6: Statistical Analysis
1. Aggregate across seeds
2. Bootstrap CIs for all metrics
3. Pairwise permutation tests between all methods
4. Cohen's d for all comparisons
5. Bonferroni correction
6. Monotonicity tests (Spearman rho)
7. Generate Pareto frontier data

**Output**: `analysis/statistical_results.json`

---

## 10. SLURM Job Structure

### 10.1 Hardware Requirements
- **GPU**: 1x B200 (180GB VRAM) -- Llama-3.1-8B-Instruct fits comfortably in fp16 (~16GB)
- **CPU**: 8 cores
- **RAM**: 64GB
- **Storage**: ~50GB for all generations and results

### 10.2 Job Decomposition

#### Job 1: Circuit Discovery (3 seeds)
```bash
#!/bin/bash
#SBATCH --job-name=v2-discovery
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=results_v2/logs/discovery_%j.log
#SBATCH --array=0-2

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -u experiments/v2_experiment.py \
    --phase discovery \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --seed $SEED \
    --top_k 200 \
    --output_dir results_v2
```
**Estimated time**: ~30 min per seed (contrastive discovery on 20+20 prompts)
**Total**: ~30 min (3 parallel jobs)

#### Job 2: Baseline + Prompt Engineering Generation
```bash
#SBATCH --time=3:00:00
#SBATCH --array=0-2

python -u experiments/v2_experiment.py \
    --phase generate \
    --methods baseline,prompt_engineering \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} \
    --n_prompts 75 \
    --max_tokens 500
```
**Estimated time**: 75 prompts x 500 tokens x 2 methods = ~150K tokens. At ~100 tok/s on B200: ~25 min per seed.

#### Job 3: Temperature Sweep
```bash
#SBATCH --time=3:00:00
#SBATCH --array=0-2

# 4 temperature values x 75 prompts x 500 tokens
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods temperature \
    --temperatures 0.3,0.5,0.7,1.0 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```
**Estimated time**: ~50 min per seed (4 sweep points)

#### Job 4: Logit Suppression Sweep
```bash
#SBATCH --time=3:00:00
#SBATCH --array=0-2

# 3 bias strengths x 75 prompts x 500 tokens
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods logit_suppression \
    --logit_biases -5,-10,-20 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```
**Estimated time**: ~40 min per seed

#### Job 5: Neuron Steering Sweep
```bash
#SBATCH --time=6:00:00
#SBATCH --array=0-2

# 14 alpha values x 75 prompts x 500 tokens
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods neuron_steering \
    --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```
**Estimated time**: ~2.5 hours per seed (14 sweep points, each needing hook setup)

#### Job 6: CAA (repeng) Sweep
```bash
#SBATCH --time=6:00:00
#SBATCH --array=0-2

# 8 alpha values x 75 prompts x 500 tokens
# Note: repeng control vector training adds ~10 min overhead
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods caa_repeng \
    --caa_alphas 0.5,1.0,1.5,2.0,3.0,5.0,7.0,10.0 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```
**Estimated time**: ~1.5 hours per seed

#### Job 7: Random Ablation Control
```bash
#SBATCH --time=4:00:00
#SBATCH --array=0-2

# 5 random samples x 75 prompts x 500 tokens
python -u experiments/v2_experiment.py \
    --phase generate \
    --methods random_ablation \
    --n_random_samples 5 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```
**Estimated time**: ~1.5 hours per seed

#### Job 8: Perplexity Measurement (all methods)
```bash
#SBATCH --time=4:00:00
#SBATCH --array=0-2

# 100 WikiText-2 passages x all method configurations
# Forward-only (no generation), much faster
python -u experiments/v2_experiment.py \
    --phase perplexity \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```
**Estimated time**: ~1 hour per seed (forward passes only, 100 passages per config)

#### Job 9: GPT-4 Judge (no GPU needed)
```bash
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

python -u experiments/v2_experiment.py \
    --phase gpt4_judge \
    --n_judge_prompts 20 \
    --n_judge_runs 3
```
**Estimated time**: ~30 min (API calls, rate-limited)

#### Job 10: Statistical Analysis (no GPU needed)
```bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

python -u experiments/v2_experiment.py \
    --phase analysis \
    --n_bootstrap 10000
```
**Estimated time**: ~20 min (bootstrap is CPU-intensive but parallelizable)

### 10.3 Job Dependencies (SLURM)
```bash
# Submit in dependency order:
JOB1=$(sbatch --parsable slurm/v2_discovery.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/v2_baseline.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/v2_temperature.sh)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/v2_logit.sh)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/v2_neuron.sh)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/v2_caa.sh)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/v2_random.sh)
JOB8=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3:$JOB4:$JOB5:$JOB6:$JOB7 slurm/v2_perplexity.sh)
JOB9=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3:$JOB4:$JOB5:$JOB6:$JOB7 slurm/v2_gpt4_judge.sh)
JOB10=$(sbatch --parsable --dependency=afterok:$JOB8:$JOB9 slurm/v2_analysis.sh)
```

### 10.4 Compute Budget Summary

| Job | GPU-hours (per seed) | GPU-hours (total, 3 seeds) | Parallelizable |
|-----|---------------------|---------------------------|----------------|
| Discovery | 0.5 | 1.5 | Yes (array) |
| Baseline + Prompt Eng | 0.4 | 1.2 | Yes (array) |
| Temperature | 0.8 | 2.4 | Yes (array) |
| Logit Suppression | 0.7 | 2.1 | Yes (array) |
| Neuron Steering | 2.5 | 7.5 | Yes (array) |
| CAA (repeng) | 1.5 | 4.5 | Yes (array) |
| Random Ablation | 1.5 | 4.5 | Yes (array) |
| Perplexity | 1.0 | 3.0 | Yes (array) |
| GPT-4 Judge | 0 (CPU) | 0 (CPU) | N/A |
| Analysis | 0 (CPU) | 0 (CPU) | N/A |
| **Total** | **~9 GPU-hr/seed** | **~27 GPU-hours** | |

**Wall-clock time with full parallelism**: ~8 hours (discovery sequential, then all generation jobs parallel, then perplexity + judge, then analysis).

**Cost estimate on cloud B200**: ~27 GPU-hours x ~$3/hr = ~$81 total. Very manageable.

---

## 11. Output Format

### 11.1 Directory Structure
```
results_v2/
  logs/                          # SLURM logs
  circuits/
    seed_42_circuit.json
    seed_123_circuit.json
    seed_456_circuit.json
    circuit_stability.json       # cross-seed Jaccard analysis
  generations/
    baseline_42.jsonl
    prompt_engineering_42.jsonl
    temperature_0.3_42.jsonl
    ...
    neuron_steering_0.0_42.jsonl
    ...
  metrics/
    per_prompt/                  # individual prompt-level metrics
    aggregated/                  # method-level aggregates
  perplexity/
    per_method/                  # ppl at each sweep point
    pareto_data.json             # (slop_reduction, ppl_ratio) pairs
  gpt4_judge/
    scores.json
    correlation.json
  analysis/
    statistical_results.json     # CIs, p-values, Cohen's d
    monotonicity.json            # dose-response analysis
    summary_table.json           # final comparison table for paper
```

### 11.2 Summary Table Format (for paper)
```
Method              | slop_density [95% CI]  | ppl_ratio [95% CI] | GPT-4 score | Cohen's d vs baseline
--------------------|------------------------|--------------------|-----------  |---------------------
Baseline            | X.XX [X.XX, X.XX]     | 1.000              | X.X         | --
Prompt engineering  | X.XX [X.XX, X.XX]     | 1.000              | X.X         | X.XX
Temperature (0.3)   | X.XX [X.XX, X.XX]     | 1.000              | X.X         | X.XX
Logit suppress (-10)| X.XX [X.XX, X.XX]     | X.XXX              | X.X         | X.XX
CAA (repeng, a=X)   | X.XX [X.XX, X.XX]     | X.XXX              | X.X         | X.XX
Neuron steer (a=X)  | X.XX [X.XX, X.XX]     | X.XXX              | X.X         | X.XX
Random ablation     | X.XX [X.XX, X.XX]     | X.XXX              | X.X         | X.XX
```

For methods with sweeps, report the "best" operating point = lowest slop_density where ppl_ratio < 1.05.

---

## 12. Possible Outcomes and Interpretation

### 12.1 Best case (neuron steering wins)
- Neuron steering achieves comparable or better slop reduction than alternatives
- With lower perplexity degradation than CAA
- And without the vocabulary-restriction artifacts of logit suppression
- Dose-response is monotonic
- Effect is specific (random ablation shows no reduction)
- **Interpretation**: Neuron-basis circuits capture style-related computation that can be precisely modulated

### 12.2 Neutral case (neuron steering ties with simpler methods)
- Logit suppression or prompt engineering achieve similar slop reduction
- Neuron steering offers no perplexity advantage
- **Interpretation**: The contribution is the circuit discovery methodology (which neurons encode style), not the steering superiority. Reframe paper around mechanistic insight, not practical tool.

### 12.3 Worst case (neuron steering doesn't work robustly)
- Non-monotonic dose-response
- Random ablation shows similar effect
- Effect disappears across seeds
- **Interpretation**: v1 results were noise or artifacts. Don't publish. Write up as negative result with honest analysis of what went wrong.

### 12.4 Integrity commitment
If the results don't support our claims, we report that honestly. The spec is designed to detect null results early (random ablation control, multi-seed stability, monotonicity tests). We do not cherry-pick seeds, operating points, or metrics after seeing results.

---

## 13. Pre-Registration Checklist

Before running:
- [ ] All 75 test prompts verified: no overlap with discovery prompts
- [ ] WikiText-2 loading verified: 100 passages with >= 50 tokens each
- [ ] repeng installed and tested on Llama-3.1-8B-Instruct
- [ ] GPT-4 API access configured
- [ ] All SLURM scripts tested with --dry-run
- [ ] This spec committed to git BEFORE any results are generated
- [ ] Random seed protocol verified: same seeds produce same circuits

---

## 14. Deferred: Second Model

The reviewer asked for multiple models. For the initial v2 submission, we use only Llama-3.1-8B-Instruct. If results are positive, we add Mistral-7B-Instruct-v0.3 or Qwen-2.5-7B-Instruct as a replication study in a follow-up. Adding a second model roughly doubles compute but is not required for addressing the statistical rigor concerns.

Rationale: the core reviewer complaint was about statistical rigor (N=10, single seed, no baselines), not model diversity. Fixing those issues on one model is more valuable than running the same underpowered experiment on two models.

---

## 15. Implementation Notes

### 15.1 Code Organization
The v2 experiment should be a single Python file (`v2_experiment.py`) with phase-based execution:
```
python v2_experiment.py --phase discovery --seed 42
python v2_experiment.py --phase generate --methods neuron_steering --seed 42 --alphas 0.0,0.5,1.0
python v2_experiment.py --phase perplexity --seed 42
python v2_experiment.py --phase gpt4_judge
python v2_experiment.py --phase analysis
```

### 15.2 Checkpointing
Each generation phase saves results incrementally (per-prompt JSONL). If a job is interrupted, it resumes from the last completed prompt. This is critical for the long neuron steering sweep (14 x 75 = 1050 generations).

### 15.3 Logging
- Use Python logging (not just print)
- Log to both stdout and file
- Record wall-clock time for every phase
- Record GPU memory usage at generation start

### 15.4 Reproducibility
- Pin all package versions in `requirements_v2.txt`
- Record exact model revision hash from HuggingFace
- Save complete config (all hyperparameters) in each output file
- Commit this spec to git before running any experiments
