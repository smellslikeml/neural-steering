"""
V2 Rigorous Anti-Slop Experiment
==================================
Addresses ALL reviewer concerns from v1 rejection:
- N=75 diverse held-out prompts (5 categories x 15)
- 3 seeds (42, 123, 456) for circuit discovery stability
- All baselines: prompt engineering, temperature, logit suppression, repeng CAA, random ablation
- WikiText-2 perplexity (100 passages)
- Bootstrap CIs, permutation tests, Cohen's d, Bonferroni
- Dose-response monotonicity tests
- Word-count normalized slop_density as primary metric

Phases (run independently via --phase):
  discovery   - Discover anti-slop circuit for each seed
  generate    - Generate on 75 test prompts for specified methods
  perplexity  - Measure perplexity on WikiText-2 for all method configs
  analysis    - Statistical analysis (bootstrap CIs, permutation tests, etc.)

See v2_rigorous_spec.md for full design.

Usage:
  python v2_experiment.py --phase discovery --seed 42
  python v2_experiment.py --phase generate --methods neuron_steering --seed 42 --alphas 0.0,0.5,1.0
  python v2_experiment.py --phase perplexity --seed 42
  python v2_experiment.py --phase analysis
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_DIR = "results_v2"
MAX_NEW_TOKENS = 500

# Seeds for multi-seed protocol
ALL_SEEDS = [42, 123, 456]

# ============================================================
# DISCOVERY PROMPTS (same as v1 — DO NOT MODIFY)
# ============================================================

DISCOVERY_TOPICS = [
    "education", "technology", "teamwork", "renewable energy", "leadership",
    "globalization", "emotional intelligence", "social media", "creativity",
    "diversity in the workplace", "artificial intelligence", "mental health",
    "climate change", "remote work", "healthcare innovation",
    "financial literacy", "urban planning", "scientific research",
    "automation", "digital privacy",
]

SLOP_POSITIVE = [
    f"Write an inspiring and comprehensive essay about the importance of {t} in today's rapidly evolving world."
    for t in DISCOVERY_TOPICS
]

SLOP_NEGATIVE = [
    f"Explain {t} like you're talking to a friend over coffee. Be blunt and specific. No filler, no clichés, no bullet lists. Short sentences. Say what you mean."
    for t in DISCOVERY_TOPICS
]

# ============================================================
# TEST PROMPTS (N=75, held-out, NO overlap with discovery)
# ============================================================

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

SUMMARIZATION = [
    "Summarize the main arguments for and against nuclear power as an energy source.",
    "Describe the key events of the Apollo 11 moon landing mission.",
    "Summarize how the internet evolved from ARPANET to the modern web.",
    "Describe the major causes and consequences of the 2008 financial crisis.",
    "Summarize the current scientific understanding of how memory works in the brain.",
    "Describe the key principles behind how electric vehicles differ from combustion engines.",
    "Summarize the history of antibiotics from penicillin to modern resistance concerns.",
    "Describe the main theories about what caused the extinction of the dinosaurs.",
    "Summarize the debate around standardized testing in schools.",
    "Describe how the global supply chain works for a typical consumer product.",
    "Summarize the scientific consensus on the health effects of processed foods.",
    "Describe the major milestones in the history of space exploration.",
    "Summarize how different countries approach universal healthcare.",
    "Describe the key factors that influence weather patterns and forecasting.",
    "Summarize the environmental impact of fast fashion.",
]

OPEN_ENDED = [
    "What makes a city feel alive?",
    "Why do people collect things?",
    "What changes when you learn a second language?",
    "Why do some friendships last decades while others fade quickly?",
    "What makes a good apology?",
    "Why is it hard to change your mind about something you believe strongly?",
    "What role does boredom play in generating new ideas?",
    "Why do people feel nostalgic for times they never experienced?",
    "What makes a house feel like a home?",
    "Why do some books stay with you long after you read them?",
    "What changes about a person when they become a parent?",
    "Why is it easier to give advice than to follow it?",
    "What makes a conversation memorable?",
    "Why do people root for underdogs?",
    "What does it mean to age well?",
]

ALL_TEST_PROMPTS = (
    CREATIVE_WRITING + TECHNICAL_EXPLANATION + ADVICE_GIVING +
    SUMMARIZATION + OPEN_ENDED
)

PROMPT_CATEGORIES = {
    "creative_writing": CREATIVE_WRITING,
    "technical_explanation": TECHNICAL_EXPLANATION,
    "advice_giving": ADVICE_GIVING,
    "summarization": SUMMARIZATION,
    "open_ended": OPEN_ENDED,
}

# ============================================================
# SLOP WORD LISTS
# ============================================================

TIER1_SLOP = {
    "delve", "utilize", "leverage", "facilitate", "elucidate", "embark",
    "endeavor", "encompass", "multifaceted", "tapestry", "testament",
    "paradigm", "synergy", "synergize", "holistic", "catalyze", "catalyst",
    "juxtapose", "nuanced", "realm", "landscape", "myriad", "plethora",
}

TIER2_SLOP = {
    "robust", "comprehensive", "seamless", "seamlessly", "cutting-edge",
    "innovative", "streamline", "empower", "foster", "enhance", "elevate",
    "optimize", "scalable", "pivotal", "intricate", "profound", "resonate",
    "underscore", "harness", "navigate", "cultivate", "bolster", "galvanize",
    "cornerstone", "game-changer",
}

TIER3_PHRASES = [
    "it's worth noting", "it's important to note", "importantly",
    "notably", "interestingly", "let's dive into", "let's explore",
    "in this section", "as we can see", "as mentioned earlier",
    "in conclusion", "to summarize", "furthermore", "moreover",
    "additionally", "in today's", "at the end of the day",
    "it goes without saying", "without further ado", "when it comes to",
    "in the realm of", "one might argue", "it could be suggested",
    "this begs the question", "great question",
    "excellent point", "absolutely",
    "in today's rapidly", "plays a crucial role", "serves as a",
    "it is essential", "it is important", "it is worth",
    "paves the way", "stands as a", "remains a", "continues to be",
    "in an era", "in a world where", "in the landscape of",
    "the importance of", "the role of", "the impact of",
    "by fostering", "by leveraging", "by embracing",
    "whether it's", "whether it is", "from enhancing",
    "a powerful tool", "a vital role", "a key component",
    "goes beyond",
]

ANTISLOP_SYSTEM_PROMPT = (
    "Write in a direct, natural style. Avoid cliches and filler phrases. "
    "Do not use bullet points or numbered lists unless specifically asked. "
    "Avoid words like: delve, tapestry, testament, paradigm, synergy, "
    "multifaceted, holistic, nuanced, realm, landscape, myriad, plethora, "
    "utilize, leverage, facilitate, encompass. "
    "Do not use em dashes. Vary your sentence length. "
    "Do not start with sycophantic phrases like 'Great question'."
)


# ============================================================
# SLOP MEASUREMENT (same as creative_circuits.py)
# ============================================================

def count_slop(text: str) -> dict:
    """Count slop markers in text. Returns detailed breakdown."""
    if not text or not text.strip():
        return {k: 0 for k in [
            "tier1_count", "tier2_count", "tier3_count", "em_dash_count",
            "not_just_but", "list_ratio", "parallel_ratio", "sentence_cv",
            "total_structural", "total_slop", "ttr", "adj_cascades",
            "colon_density", "start_diversity", "composite_slop", "word_count",
            "slop_density",
        ]}
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_set = set(words)

    tier1_hits = word_set & TIER1_SLOP
    tier2_hits = word_set & TIER2_SLOP
    tier3_hits = [p for p in TIER3_PHRASES if p in text_lower]

    tier1_count = sum(1 for w in words if w in TIER1_SLOP)
    tier2_count = sum(1 for w in words if w in TIER2_SLOP)

    em_dash_count = text.count("\u2014") + text.count(" -- ")
    not_just_but = len(re.findall(r"not just\b.*?\bbut\b", text_lower))

    lines = text.strip().split("\n")
    bullet_lines = sum(1 for line in lines if re.match(r'\s*[-*\u2022]\s', line) or re.match(r'\s*\d+\.\s', line))
    list_ratio = bullet_lines / max(len(lines), 1)

    line_starts = []
    for line in lines:
        stripped = line.strip().lstrip("-*\u20220123456789. ")
        first_word = stripped.split()[0].lower() if stripped.split() else ""
        line_starts.append(first_word)
    if len(line_starts) > 2:
        start_counts = Counter(line_starts)
        most_common_start = start_counts.most_common(1)[0][1] if start_counts else 0
        parallel_ratio = most_common_start / len(line_starts)
    else:
        parallel_ratio = 0

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]
    sent_lengths = [len(s.split()) for s in sentences]
    if len(sent_lengths) > 2:
        sent_cv = np.std(sent_lengths) / max(np.mean(sent_lengths), 1)
    else:
        sent_cv = 0

    syco_openers = ["great question", "excellent point", "that's a great",
                    "absolutely", "you raise an important", "what a great"]
    has_syco_opening = any(text_lower.strip().startswith(s) for s in syco_openers)

    total_structural = em_dash_count + not_just_but + (1 if has_syco_opening else 0)
    total_slop = tier1_count + tier2_count + len(tier3_hits) + total_structural

    unique = set(words)
    ttr = len(unique) / max(len(words), 1)

    adj_cascades = len(re.findall(r'\b\w+,\s+\w+(?:,\s+(?:and\s+)?|\s+and\s+)\w+\b', text))
    colon_count = text.count(":") + text.count(";")

    sentence_starts = []
    for s in sentences:
        ws = s.strip().split()
        if len(ws) >= 2:
            sentence_starts.append(f"{ws[0].lower()} {ws[1].lower()}")
    start_diversity = len(set(sentence_starts)) / max(len(sentence_starts), 1)

    composite = (
        (tier1_count * 3.0) +
        (tier2_count * 1.5) +
        (len(tier3_hits) * 2.0) +
        (em_dash_count * 1.0) +
        (not_just_but * 2.0) +
        (1.0 if has_syco_opening else 0) +
        (list_ratio * 3.0) +
        (adj_cascades * 2.0) +
        (max(0, 0.4 - sent_cv) * 5.0 if len(sent_lengths) > 3 else 0) +
        max(0, 0.5 - ttr) * 3.0
    )

    return {
        "tier1_count": tier1_count,
        "tier2_count": tier2_count,
        "tier3_count": len(tier3_hits),
        "em_dash_count": em_dash_count,
        "not_just_but": not_just_but,
        "list_ratio": round(list_ratio, 3),
        "parallel_ratio": round(parallel_ratio, 3),
        "sentence_cv": round(float(sent_cv), 3),
        "syco_opening": has_syco_opening,
        "total_structural": total_structural,
        "total_slop": total_slop,
        "ttr": round(ttr, 4),
        "adj_cascades": adj_cascades,
        "colon_density": round(colon_count / max(len(words), 1), 4),
        "start_diversity": round(start_diversity, 4),
        "composite_slop": round(composite, 3),
        "word_count": len(words),
        "slop_density": round(total_slop / max(len(words), 1), 6),
    }


# ============================================================
# STATISTICAL UTILITIES
# ============================================================

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
    return float(statistic(data)), float(lower), float(upper)


def paired_permutation_test(x, y, n_permutations=10000, seed=42):
    """Two-sided paired permutation test. H0: no difference in means."""
    rng = np.random.RandomState(seed)
    diff = np.array(x) - np.array(y)
    observed = np.abs(np.mean(diff))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_diff = np.abs(np.mean(diff * signs))
        if perm_diff >= observed:
            count += 1
    return count / n_permutations


def cohens_d(x, y):
    """Paired Cohen's d effect size."""
    diff = np.array(x) - np.array(y)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def spearman_rho(x, y):
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


# ============================================================
# HELPER: SET ALL SEEDS
# ============================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# HELPER: VALIDATE NO OVERLAP
# ============================================================

def validate_no_overlap():
    """Ensure test prompts don't contain discovery topics."""
    for prompt in ALL_TEST_PROMPTS:
        prompt_lower = prompt.lower()
        for topic in DISCOVERY_TOPICS:
            if topic in prompt_lower:
                raise ValueError(
                    f"Test prompt contains discovery topic '{topic}': {prompt}"
                )
    log.info("Validated: zero overlap between test prompts and discovery topics")


# ============================================================
# HELPER: LOAD MODEL + TOKENIZER
# ============================================================

def load_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    log.info(f"Model loaded. Device: {model.device}, dtype: {model.dtype}")
    return model, tokenizer


# ============================================================
# HELPER: GENERATE TEXT
# ============================================================

def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                  system_prompt=None, do_sample=False, temperature=1.0,
                  top_p=0.95):
    """Generate text using chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ============================================================
# HELPER: PERPLEXITY COMPUTATION
# ============================================================

def compute_perplexity(model, tokenizer, texts, max_length=256):
    """Standard perplexity on a list of texts."""
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=max_length)
        input_ids = encodings["input_ids"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        nll = outputs.loss.item() * (input_ids.shape[1] - 1)
        total_nll += nll
        total_tokens += input_ids.shape[1] - 1
    return math.exp(total_nll / total_tokens)


def load_wikitext_passages(n=100, min_words=50):
    """Load first n qualifying passages from WikiText-2 test set."""
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    passages = []
    for text in wt["text"]:
        text = text.strip()
        if len(text.split()) >= min_words:
            passages.append(text)
        if len(passages) >= n:
            break
    log.info(f"Loaded {len(passages)} WikiText-2 passages (min {min_words} words)")
    return passages


# ============================================================
# HELPER: LOGIT SUPPRESSION PROCESSOR
# ============================================================

def build_slop_token_ids(tokenizer):
    """Get all token IDs for slop words (with/without leading space, capitalized)."""
    slop_words = list(TIER1_SLOP) + list(TIER2_SLOP)
    token_ids = set()
    for word in slop_words:
        for variant in [word, word.capitalize(), f" {word}", f" {word.capitalize()}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            token_ids.update(ids)
    log.info(f"Built slop token suppression set: {len(token_ids)} unique token IDs from {len(slop_words)} words")
    return sorted(token_ids)


class SlopLogitsProcessor:
    """Suppress slop token logits by a fixed bias."""
    def __init__(self, token_ids, bias):
        self.token_ids = token_ids
        self.bias = bias

    def __call__(self, input_ids, scores):
        for tid in self.token_ids:
            if tid < scores.shape[-1]:
                scores[:, tid] += self.bias
        return scores


# ============================================================
# PHASE: DISCOVERY
# ============================================================

def phase_discovery(args):
    """Discover anti-slop circuit for a given seed."""
    from neuron_steer.core import NeuronSteerer

    seed = args.seed
    set_all_seeds(seed)
    validate_no_overlap()

    # NeuronSteerer loads model internally (with eager attention for LRP)
    steerer = NeuronSteerer(args.model)

    log.info(f"=== Circuit Discovery (seed={seed}) ===")
    circuit = steerer.discover_contrastive(
        SLOP_POSITIVE, SLOP_NEGATIVE,
        top_k=args.top_k,
    )

    # Save circuit
    out_dir = Path(args.output_dir) / "circuits"
    out_dir.mkdir(parents=True, exist_ok=True)

    circuit_data = {
        "seed": seed,
        "top_k": args.top_k,
        "n_neurons": len(circuit.neurons),
        "neurons": {
            f"L{n.layer}_N{n.neuron}": float(attr)
            for n, attr in circuit.neurons.items()
        },
        "layer_distribution": {},
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Layer distribution
    from collections import Counter
    layers = [n.layer for n in circuit.neurons.keys()]
    layer_counts = Counter(layers)
    circuit_data["layer_distribution"] = {str(k): v for k, v in sorted(layer_counts.items())}

    out_path = out_dir / f"seed_{seed}_circuit.json"
    with open(out_path, "w") as f:
        json.dump(circuit_data, f, indent=2)
    log.info(f"Saved circuit to {out_path}: {len(circuit.neurons)} neurons")

    return circuit_data


# ============================================================
# PHASE: GENERATE
# ============================================================

def phase_generate(args):
    """Generate on 75 test prompts for specified methods."""
    from neuron_steer.core import NeuronSteerer, NeuronIdx, Circuit, steer_neurons
    from transformers import LogitsProcessorList

    seed = args.seed
    set_all_seeds(seed)
    validate_no_overlap()

    model, tokenizer = load_model(args.model)

    out_dir = Path(args.output_dir) / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods.split(",")
    log.info(f"Methods to run: {methods}")

    # Load circuit if needed
    circuit = None
    circuit_neurons = None
    if any(m in methods for m in ["neuron_steering", "random_ablation"]):
        circuit_path = Path(args.output_dir) / "circuits" / f"seed_{seed}_circuit.json"
        if not circuit_path.exists():
            raise FileNotFoundError(f"Circuit not found: {circuit_path}. Run --phase discovery first.")
        with open(circuit_path) as f:
            circuit_data = json.load(f)
        # Reconstruct neurons dict
        circuit_neurons = {}
        for key, attr in circuit_data["neurons"].items():
            parts = key.replace("L", "").replace("N", "").split("_")
            layer, neuron = int(parts[0]), int(parts[1])
            nidx = NeuronIdx(layer=layer, position=-1, neuron=neuron)
            circuit_neurons[nidx] = attr
        circuit = Circuit(
            neurons=circuit_neurons,
            prompt="contrastive",
            target_token="slop",
            total_logit_diff=0.0,
        )
        log.info(f"Loaded circuit: {len(circuit_neurons)} neurons")

    # Pre-build slop token IDs for logit suppression
    slop_token_ids = None
    if "logit_suppression" in methods:
        slop_token_ids = build_slop_token_ids(tokenizer)

    # --- GENERATE FOR EACH METHOD ---

    for method in methods:
        log.info(f"\n{'='*60}")
        log.info(f"METHOD: {method}")
        log.info(f"{'='*60}")

        if method == "baseline":
            _generate_baseline(model, tokenizer, seed, out_dir)

        elif method == "prompt_engineering":
            _generate_prompt_engineering(model, tokenizer, seed, out_dir)

        elif method == "temperature":
            temps = [float(t) for t in args.temperatures.split(",")]
            _generate_temperature(model, tokenizer, seed, out_dir, temps)

        elif method == "logit_suppression":
            biases = [float(b) for b in args.logit_biases.split(",")]
            _generate_logit_suppression(model, tokenizer, seed, out_dir,
                                        slop_token_ids, biases)

        elif method == "neuron_steering":
            alphas = [float(a) for a in args.alphas.split(",")]
            _generate_neuron_steering(model, tokenizer, seed, out_dir,
                                      circuit, alphas)

        elif method == "caa_repeng":
            caa_alphas = [float(a) for a in args.caa_alphas.split(",")]
            _generate_caa_repeng(model, tokenizer, seed, out_dir, caa_alphas)

        elif method == "random_ablation":
            _generate_random_ablation(model, tokenizer, seed, out_dir,
                                      circuit, args.n_random_samples)

        else:
            log.warning(f"Unknown method: {method}")


def _run_generation_loop(model, tokenizer, prompts, out_path, gen_fn, method_label):
    """Common generation loop with checkpointing."""
    # Check for existing checkpoint
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing.add(rec["prompt_idx"])
        log.info(f"  Resuming: {len(existing)} prompts already done")

    with open(out_path, "a") as f:
        for i, prompt in enumerate(prompts):
            if i in existing:
                continue
            t0 = time.time()
            text = gen_fn(prompt)
            elapsed = time.time() - t0
            slop = count_slop(text)
            rec = {
                "prompt_idx": i,
                "prompt": prompt,
                "category": _get_category(i),
                "generation": text,
                "slop_metrics": slop,
                "elapsed_sec": round(elapsed, 2),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                log.info(f"  [{method_label}] {i+1}/{len(prompts)} | "
                         f"slop_density={slop['slop_density']:.4f} | "
                         f"composite={slop['composite_slop']:.1f}")


def _get_category(prompt_idx):
    """Map prompt index to category name."""
    cats = list(PROMPT_CATEGORIES.keys())
    return cats[prompt_idx // 15] if prompt_idx < 75 else "unknown"


def _generate_baseline(model, tokenizer, seed, out_dir):
    out_path = out_dir / f"baseline_{seed}.jsonl"
    def gen_fn(prompt):
        return generate_text(model, tokenizer, prompt, do_sample=False)
    _run_generation_loop(model, tokenizer, ALL_TEST_PROMPTS, out_path, gen_fn, "baseline")


def _generate_prompt_engineering(model, tokenizer, seed, out_dir):
    out_path = out_dir / f"prompt_engineering_{seed}.jsonl"
    def gen_fn(prompt):
        return generate_text(model, tokenizer, prompt, do_sample=False,
                             system_prompt=ANTISLOP_SYSTEM_PROMPT)
    _run_generation_loop(model, tokenizer, ALL_TEST_PROMPTS, out_path, gen_fn, "prompt_eng")


def _generate_temperature(model, tokenizer, seed, out_dir, temps):
    for temp in temps:
        out_path = out_dir / f"temperature_{temp}_{seed}.jsonl"
        def gen_fn(prompt, t=temp):
            return generate_text(model, tokenizer, prompt, do_sample=True,
                                 temperature=t, top_p=0.95)
        _run_generation_loop(model, tokenizer, ALL_TEST_PROMPTS, out_path, gen_fn,
                             f"temp={temp}")


def _generate_logit_suppression(model, tokenizer, seed, out_dir, token_ids, biases):
    from transformers import LogitsProcessorList

    for bias in biases:
        out_path = out_dir / f"logit_suppression_{bias}_{seed}.jsonl"
        processor = SlopLogitsProcessor(token_ids, bias)

        def gen_fn(prompt, p=processor):
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=LogitsProcessorList([p]),
                )
            generated = output[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated, skip_special_tokens=True)

        _run_generation_loop(model, tokenizer, ALL_TEST_PROMPTS, out_path, gen_fn,
                             f"logit_bias={bias}")


def _generate_neuron_steering(model, tokenizer, seed, out_dir, circuit, alphas):
    from neuron_steer.core import steer_neurons

    for alpha in alphas:
        out_path = out_dir / f"neuron_steering_{alpha}_{seed}.jsonl"

        if alpha == 1.0:
            # No intervention (same as baseline, but we record it for completeness)
            def gen_fn(prompt):
                return generate_text(model, tokenizer, prompt, do_sample=False)
        else:
            def gen_fn(prompt, a=alpha):
                with steer_neurons(model, circuit.neurons, multiplier=a):
                    return generate_text(model, tokenizer, prompt, do_sample=False)

        _run_generation_loop(model, tokenizer, ALL_TEST_PROMPTS, out_path, gen_fn,
                             f"neuron_a={alpha}")


def _generate_caa_repeng(model, tokenizer, seed, out_dir, caa_alphas):
    """Generate using repeng control vectors."""
    from repeng import ControlVector, ControlModel
    from repeng.extract import DatasetEntry

    log.info("Training repeng control vector...")

    # Build dataset: pair each positive with corresponding negative
    dataset = []
    for pos, neg in zip(SLOP_POSITIVE, SLOP_NEGATIVE):
        dataset.append(DatasetEntry(positive=pos, negative=neg))

    # Wrap model in ControlModel (middle layers, standard repeng approach)
    n_layers = model.config.num_hidden_layers
    # repeng standard: use layers -1 to -(n_layers-1), effectively all except first
    # Common pattern: skip first and last few layers
    layer_ids = list(range(1, n_layers - 1))
    log.info(f"Creating ControlModel with layers {layer_ids[0]}-{layer_ids[-1]} "
             f"({len(layer_ids)} layers)")

    ctrl_model = ControlModel(model, layer_ids)
    cv = ControlVector.train(ctrl_model, tokenizer, dataset)
    log.info("Control vector trained successfully")

    for alpha in caa_alphas:
        out_path = out_dir / f"caa_repeng_{alpha}_{seed}.jsonl"

        def gen_fn(prompt, a=alpha):
            # Apply control vector (negative coeff = reduce the positive direction = reduce slop)
            ctrl_model.set_control(cv, coeff=-a)
            try:
                return generate_text(ctrl_model, tokenizer, prompt, do_sample=False)
            finally:
                ctrl_model.reset()

        _run_generation_loop(model, tokenizer, ALL_TEST_PROMPTS, out_path, gen_fn,
                             f"caa_a={alpha}")

    # Unwrap model after all CAA sweeps
    ctrl_model.unwrap()
    log.info("ControlModel unwrapped")


def _generate_random_ablation(model, tokenizer, seed, out_dir, circuit, n_samples):
    """Random ablation with matched layer distribution."""
    from neuron_steer.core import NeuronIdx, Circuit, steer_neurons

    # Get layer distribution from real circuit
    layer_counts = Counter(n.layer for n in circuit.neurons.keys())
    total_neurons = len(circuit.neurons)
    n_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size

    for sample_idx in range(n_samples):
        out_path = out_dir / f"random_ablation_sample{sample_idx}_{seed}.jsonl"

        # Generate random circuit with same layer distribution
        rng = random.Random(seed * 1000 + sample_idx)
        random_neurons = {}
        for layer, count in layer_counts.items():
            for _ in range(count):
                neuron = rng.randint(0, intermediate_size - 1)
                nidx = NeuronIdx(layer=layer, position=-1, neuron=neuron)
                random_neurons[nidx] = 0.0  # attribution doesn't matter for ablation

        random_circuit = Circuit(
            neurons=random_neurons,
            prompt="random",
            target_token="random",
            total_logit_diff=0.0,
        )

        def gen_fn(prompt, rc=random_circuit):
            with steer_neurons(model, rc.neurons, multiplier=0.0):
                return generate_text(model, tokenizer, prompt, do_sample=False)

        _run_generation_loop(model, tokenizer, ALL_TEST_PROMPTS, out_path, gen_fn,
                             f"random_sample{sample_idx}")


# ============================================================
# PHASE: PERPLEXITY
# ============================================================

def phase_perplexity(args):
    """Measure perplexity for all method configurations."""
    from neuron_steer.core import NeuronSteerer, NeuronIdx, Circuit, steer_neurons

    seed = args.seed
    set_all_seeds(seed)

    model, tokenizer = load_model(args.model)
    passages = load_wikitext_passages(n=100, min_words=50)

    out_dir = Path(args.output_dir) / "perplexity"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Baseline ---
    log.info("Perplexity: baseline")
    baseline_ppl = compute_perplexity(model, tokenizer, passages)
    results["baseline"] = {"ppl": baseline_ppl, "ppl_ratio": 1.0}
    log.info(f"  Baseline ppl = {baseline_ppl:.4f}")

    # --- Neuron steering sweep ---
    circuit_path = Path(args.output_dir) / "circuits" / f"seed_{seed}_circuit.json"
    if circuit_path.exists():
        with open(circuit_path) as f:
            circuit_data = json.load(f)
        circuit_neurons = {}
        for key, attr in circuit_data["neurons"].items():
            parts = key.replace("L", "").replace("N", "").split("_")
            layer, neuron = int(parts[0]), int(parts[1])
            nidx = NeuronIdx(layer=layer, position=-1, neuron=neuron)
            circuit_neurons[nidx] = attr

        alphas = [float(a) for a in args.alphas.split(",")]
        for alpha in alphas:
            label = f"neuron_steering_{alpha}"
            log.info(f"Perplexity: {label}")
            if alpha == 1.0:
                ppl = baseline_ppl
            else:
                with steer_neurons(model, circuit_neurons, multiplier=alpha):
                    ppl = compute_perplexity(model, tokenizer, passages)
            results[label] = {"ppl": ppl, "ppl_ratio": ppl / baseline_ppl}
            log.info(f"  {label}: ppl={ppl:.4f}, ratio={ppl/baseline_ppl:.4f}")

        # --- Random ablation ---
        n_samples = args.n_random_samples
        layer_counts = Counter(n.layer for n in circuit_neurons.keys())
        intermediate_size = model.config.intermediate_size

        for sample_idx in range(n_samples):
            label = f"random_ablation_sample{sample_idx}"
            log.info(f"Perplexity: {label}")
            rng = random.Random(seed * 1000 + sample_idx)
            random_neurons = {}
            for layer_id, count in layer_counts.items():
                for _ in range(count):
                    neuron_id = rng.randint(0, intermediate_size - 1)
                    nidx = NeuronIdx(layer=layer_id, position=-1, neuron=neuron_id)
                    random_neurons[nidx] = 0.0
            with steer_neurons(model, random_neurons, multiplier=0.0):
                ppl = compute_perplexity(model, tokenizer, passages)
            results[label] = {"ppl": ppl, "ppl_ratio": ppl / baseline_ppl}
            log.info(f"  {label}: ppl={ppl:.4f}, ratio={ppl/baseline_ppl:.4f}")

    # --- CAA (repeng) ---
    caa_alphas = [float(a) for a in args.caa_alphas.split(",")]
    if caa_alphas:
        from repeng import ControlVector, ControlModel
        from repeng.extract import DatasetEntry

        dataset = [DatasetEntry(positive=pos, negative=neg)
                   for pos, neg in zip(SLOP_POSITIVE, SLOP_NEGATIVE)]
        n_layers = model.config.num_hidden_layers
        layer_ids = list(range(1, n_layers - 1))
        ctrl_model = ControlModel(model, layer_ids)
        cv = ControlVector.train(ctrl_model, tokenizer, dataset)

        for alpha in caa_alphas:
            label = f"caa_repeng_{alpha}"
            log.info(f"Perplexity: {label}")
            ctrl_model.set_control(cv, coeff=-alpha)
            ppl = compute_perplexity(ctrl_model, tokenizer, passages)
            ctrl_model.reset()
            results[label] = {"ppl": ppl, "ppl_ratio": ppl / baseline_ppl}
            log.info(f"  {label}: ppl={ppl:.4f}, ratio={ppl/baseline_ppl:.4f}")

        ctrl_model.unwrap()

    # --- Logit suppression ---
    biases = [float(b) for b in args.logit_biases.split(",")]
    if biases:
        # Logit suppression doesn't affect forward pass (only generation),
        # so perplexity = baseline. Record for completeness.
        for bias in biases:
            label = f"logit_suppression_{bias}"
            results[label] = {"ppl": baseline_ppl, "ppl_ratio": 1.0,
                              "note": "logit suppression does not affect forward-pass perplexity"}

    # --- Temperature ---
    # Temperature doesn't affect greedy forward pass perplexity
    temps = [float(t) for t in args.temperatures.split(",")]
    for temp in temps:
        label = f"temperature_{temp}"
        results[label] = {"ppl": baseline_ppl, "ppl_ratio": 1.0,
                          "note": "temperature does not affect forward-pass perplexity"}

    # --- Prompt engineering ---
    results["prompt_engineering"] = {"ppl": baseline_ppl, "ppl_ratio": 1.0,
                                     "note": "system prompt does not affect forward-pass perplexity"}

    # Save results
    out_path = out_dir / f"perplexity_seed_{seed}.json"
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "baseline_ppl": baseline_ppl, "n_passages": 100,
                    "results": results}, f, indent=2)
    log.info(f"\nSaved perplexity results to {out_path}")


# ============================================================
# PHASE: ANALYSIS
# ============================================================

def phase_analysis(args):
    """Statistical analysis: bootstrap CIs, permutation tests, etc."""
    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generations"
    ppl_dir = out_dir / "perplexity"
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all generation files
    gen_files = sorted(gen_dir.glob("*.jsonl"))
    log.info(f"Found {len(gen_files)} generation files")

    # Parse method/seed/param from filenames
    # Format: {method}_{param}_{seed}.jsonl or {method}_{seed}.jsonl
    method_data = {}  # method_key -> {seed -> [slop_density per prompt]}

    for gf in gen_files:
        records = []
        with open(gf) as f:
            for line in f:
                records.append(json.loads(line))

        # Parse filename
        stem = gf.stem  # e.g. "neuron_steering_0.4_42"
        # Extract seed (last part)
        parts = stem.rsplit("_", 1)
        try:
            file_seed = int(parts[-1])
        except ValueError:
            log.warning(f"Cannot parse seed from {gf.name}, skipping")
            continue
        method_key = parts[0]

        slop_densities = [r["slop_metrics"]["slop_density"] for r in records]
        composite_slops = [r["slop_metrics"]["composite_slop"] for r in records]
        word_counts = [r["slop_metrics"]["word_count"] for r in records]

        if method_key not in method_data:
            method_data[method_key] = {}
        method_data[method_key][file_seed] = {
            "slop_density": slop_densities,
            "composite_slop": composite_slops,
            "word_count": word_counts,
            "n_prompts": len(records),
        }

    log.info(f"Methods found: {sorted(method_data.keys())}")

    # --- Aggregate across seeds ---
    summary = {}
    for method_key, seed_data in sorted(method_data.items()):
        all_densities = []
        all_composites = []
        for seed_val, data in seed_data.items():
            all_densities.extend(data["slop_density"])
            all_composites.extend(data["composite_slop"])

        all_densities = np.array(all_densities)
        all_composites = np.array(all_composites)

        mean_d, lo_d, hi_d = bootstrap_ci(all_densities, n_bootstrap=args.n_bootstrap)
        mean_c, lo_c, hi_c = bootstrap_ci(all_composites, n_bootstrap=args.n_bootstrap)

        summary[method_key] = {
            "n_observations": len(all_densities),
            "n_seeds": len(seed_data),
            "slop_density": {"mean": mean_d, "ci_lower": lo_d, "ci_upper": hi_d},
            "composite_slop": {"mean": mean_c, "ci_lower": lo_c, "ci_upper": hi_c},
        }
        log.info(f"{method_key:40s}: slop_density={mean_d:.4f} [{lo_d:.4f}, {hi_d:.4f}]  "
                 f"composite={mean_c:.2f} [{lo_c:.2f}, {hi_c:.2f}]")

    # --- Pairwise comparisons (vs baseline) ---
    baseline_key = "baseline"
    if baseline_key in method_data:
        baseline_densities = []
        for seed_val, data in method_data[baseline_key].items():
            baseline_densities.extend(data["slop_density"])
        baseline_densities = np.array(baseline_densities)

        pairwise = {}
        for method_key, seed_data in sorted(method_data.items()):
            if method_key == baseline_key:
                continue
            method_densities = []
            for seed_val, data in seed_data.items():
                method_densities.extend(data["slop_density"])
            method_densities = np.array(method_densities)

            # Ensure same length (may differ if jobs incomplete)
            min_len = min(len(baseline_densities), len(method_densities))
            if min_len == 0:
                continue
            bd = baseline_densities[:min_len]
            md = method_densities[:min_len]

            p_val = paired_permutation_test(bd, md, n_permutations=args.n_bootstrap)
            d_val = cohens_d(bd, md)
            reduction_pct = (np.mean(bd) - np.mean(md)) / np.mean(bd) * 100

            pairwise[method_key] = {
                "p_value": p_val,
                "cohens_d": round(d_val, 4),
                "reduction_pct": round(reduction_pct, 2),
                "n_paired": min_len,
            }
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            log.info(f"  vs {method_key:35s}: d={d_val:+.3f}, reduction={reduction_pct:+.1f}%, "
                     f"p={p_val:.4f} {sig}")

        summary["pairwise_vs_baseline"] = pairwise

    # --- Bonferroni correction ---
    n_comparisons = len([k for k in method_data if k != baseline_key])
    bonferroni_threshold = 0.05 / max(n_comparisons, 1)
    summary["bonferroni_threshold"] = bonferroni_threshold
    summary["n_comparisons"] = n_comparisons
    log.info(f"\nBonferroni threshold: {bonferroni_threshold:.5f} ({n_comparisons} comparisons)")

    # --- Dose-response monotonicity for neuron steering ---
    neuron_keys = sorted([k for k in method_data if k.startswith("neuron_steering_")])
    if len(neuron_keys) >= 3:
        alphas_vals = []
        densities_vals = []
        for k in neuron_keys:
            alpha = float(k.replace("neuron_steering_", ""))
            mean_d = np.mean([d for sd in method_data[k].values() for d in sd["slop_density"]])
            alphas_vals.append(alpha)
            densities_vals.append(mean_d)

        try:
            rho, p_mono = spearman_rho(alphas_vals, densities_vals)
            summary["neuron_steering_monotonicity"] = {
                "spearman_rho": round(rho, 4),
                "p_value": round(p_mono, 6),
                "alphas": alphas_vals,
                "mean_slop_densities": [round(d, 6) for d in densities_vals],
                "is_monotonic": rho > 0,  # higher alpha (less ablation) → more slop
            }
            log.info(f"\nNeuron steering monotonicity: rho={rho:.4f}, p={p_mono:.6f}")
        except Exception as e:
            log.warning(f"Could not compute monotonicity: {e}")

    # --- CAA monotonicity ---
    caa_keys = sorted([k for k in method_data if k.startswith("caa_repeng_")])
    if len(caa_keys) >= 3:
        caa_alphas_vals = []
        caa_densities_vals = []
        for k in caa_keys:
            alpha = float(k.replace("caa_repeng_", ""))
            mean_d = np.mean([d for sd in method_data[k].values() for d in sd["slop_density"]])
            caa_alphas_vals.append(alpha)
            caa_densities_vals.append(mean_d)

        try:
            rho, p_mono = spearman_rho(caa_alphas_vals, caa_densities_vals)
            summary["caa_monotonicity"] = {
                "spearman_rho": round(rho, 4),
                "p_value": round(p_mono, 6),
                "alphas": caa_alphas_vals,
                "mean_slop_densities": [round(d, 6) for d in caa_densities_vals],
            }
            log.info(f"CAA monotonicity: rho={rho:.4f}, p={p_mono:.6f}")
        except Exception as e:
            log.warning(f"Could not compute CAA monotonicity: {e}")

    # --- Load perplexity results ---
    ppl_files = sorted(ppl_dir.glob("perplexity_seed_*.json"))
    if ppl_files:
        ppl_results = {}
        for pf in ppl_files:
            with open(pf) as f:
                ppl_data = json.load(f)
            for method_key, vals in ppl_data["results"].items():
                if method_key not in ppl_results:
                    ppl_results[method_key] = []
                ppl_results[method_key].append(vals)

        summary["perplexity"] = {}
        for method_key, vals_list in sorted(ppl_results.items()):
            ppls = [v["ppl"] for v in vals_list]
            ratios = [v["ppl_ratio"] for v in vals_list]
            summary["perplexity"][method_key] = {
                "mean_ppl": round(np.mean(ppls), 4),
                "std_ppl": round(np.std(ppls, ddof=1), 4) if len(ppls) > 1 else 0,
                "mean_ratio": round(np.mean(ratios), 4),
            }

    # --- Circuit stability across seeds ---
    circuit_dir = out_dir / "circuits"
    circuit_files = sorted(circuit_dir.glob("seed_*_circuit.json"))
    if len(circuit_files) >= 2:
        circuits = {}
        for cf in circuit_files:
            with open(cf) as f:
                cd = json.load(f)
            seed_val = cd["seed"]
            circuits[seed_val] = set(cd["neurons"].keys())

        stability = {}
        seeds = sorted(circuits.keys())
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                s1, s2 = seeds[i], seeds[j]
                intersection = len(circuits[s1] & circuits[s2])
                union = len(circuits[s1] | circuits[s2])
                jaccard = intersection / max(union, 1)
                stability[f"seed_{s1}_vs_{s2}"] = {
                    "jaccard": round(jaccard, 4),
                    "intersection": intersection,
                    "union": union,
                    "size_1": len(circuits[s1]),
                    "size_2": len(circuits[s2]),
                }
                log.info(f"Circuit stability: seed {s1} vs {s2}: "
                         f"Jaccard={jaccard:.4f} ({intersection}/{union})")

        summary["circuit_stability"] = stability

    # --- Save ---
    out_path = analysis_dir / "statistical_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSaved analysis to {out_path}")

    # --- Print summary table ---
    log.info("\n" + "=" * 100)
    log.info("SUMMARY TABLE")
    log.info("=" * 100)
    log.info(f"{'Method':<40} {'slop_density [95% CI]':>30} {'ppl_ratio':>12} {'Cohen d':>10} {'p-value':>10}")
    log.info("-" * 100)
    for method_key in sorted(summary.keys()):
        if method_key in ["pairwise_vs_baseline", "bonferroni_threshold", "n_comparisons",
                          "perplexity", "circuit_stability", "neuron_steering_monotonicity",
                          "caa_monotonicity"]:
            continue
        s = summary[method_key]
        sd = s["slop_density"]
        ci_str = f"{sd['mean']:.4f} [{sd['ci_lower']:.4f}, {sd['ci_upper']:.4f}]"
        ppl_str = ""
        if "perplexity" in summary and method_key in summary["perplexity"]:
            ppl_str = f"{summary['perplexity'][method_key]['mean_ratio']:.4f}"
        d_str = ""
        p_str = ""
        if "pairwise_vs_baseline" in summary and method_key in summary["pairwise_vs_baseline"]:
            pw = summary["pairwise_vs_baseline"][method_key]
            d_str = f"{pw['cohens_d']:+.3f}"
            p_str = f"{pw['p_value']:.4f}"
        log.info(f"{method_key:<40} {ci_str:>30} {ppl_str:>12} {d_str:>10} {p_str:>10}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V2 Rigorous Anti-Slop Experiment")
    parser.add_argument("--phase", required=True,
                        choices=["discovery", "generate", "perplexity", "analysis"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=200)

    # Generation params
    parser.add_argument("--methods", default="baseline",
                        help="Comma-separated: baseline,prompt_engineering,temperature,"
                             "logit_suppression,neuron_steering,caa_repeng,random_ablation")
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0")
    parser.add_argument("--caa_alphas", default="0.5,1.0,1.5,2.0,3.0,5.0,7.0,10.0")
    parser.add_argument("--temperatures", default="0.3,0.5,0.7,1.0")
    parser.add_argument("--logit_biases", default="-5,-10,-20")
    parser.add_argument("--n_random_samples", type=int, default=5)

    # Analysis params
    parser.add_argument("--n_bootstrap", type=int, default=10000)

    args = parser.parse_args()

    log.info(f"V2 Rigorous Anti-Slop Experiment")
    log.info(f"Phase: {args.phase}")
    log.info(f"Model: {args.model}")
    log.info(f"Seed: {args.seed}")
    log.info(f"Output: {args.output_dir}")

    t0 = time.time()

    if args.phase == "discovery":
        phase_discovery(args)
    elif args.phase == "generate":
        phase_generate(args)
    elif args.phase == "perplexity":
        phase_perplexity(args)
    elif args.phase == "analysis":
        phase_analysis(args)

    elapsed = time.time() - t0
    log.info(f"\nPhase '{args.phase}' completed in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
