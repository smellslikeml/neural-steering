"""
Experiment: Belief/Opinion Steering via Neuron-Basis Circuit Discovery

Research question: Can we discover a compact neuron circuit that causally
controls whether the model affirms or denies a belief, independent of
factual knowledge? And does steering this circuit preserve unrelated
factual capabilities?

Method:
  1. Build contrastive prompt sets: one where the model affirms a stance,
     one where it denies/rejects it.
  2. Use discover_contrastive() to find the "belief affirmation" circuit.
  3. Steer at multipliers [0.0, 0.5, 1.0, 1.5, 2.0, 3.0] and measure:
     a) Affirmation rate on held-out belief prompts (does steering flip stance?)
     b) Factual accuracy on unrelated knowledge probes (specificity)
     c) Generation quality via perplexity (coherence)
  4. Report circuit statistics (layer distribution, size, overlap with refusal)

This is original work: Arora et al. demonstrated factual (capitals) and
refusal circuits. We extend to subjective belief manipulation and test
whether belief circuits are independent of factual knowledge circuits.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch

from neuron_steer.core import NeuronSteerer, Circuit


# ============================================================
# Prompt datasets
# ============================================================

# POSITIVE SET: Prompts where the model should AFFIRM a belief
# These are structured to elicit agreement, validation, or endorsement
BELIEF_AFFIRM_PROMPTS = [
    # AI-related beliefs
    "Do you think AI will be beneficial for humanity? I believe AI will greatly benefit humanity because",
    "Is artificial intelligence a positive development? Yes, AI is overwhelmingly positive because",
    "Will AI make the world better? Absolutely, AI will improve the world by",
    "Should we embrace AI technology? We should definitely embrace AI because",
    "Is machine learning going to help solve major problems? Yes, machine learning will solve",
    # Climate/science beliefs
    "Is renewable energy the future? Yes, renewable energy is clearly the future because",
    "Should we invest more in space exploration? Absolutely, space exploration is worth investing in because",
    "Is science education important? Yes, science education is critically important because",
    # Technology optimism
    "Will technology solve our biggest challenges? Yes, technology will solve our challenges because",
    "Is the internet a net positive for society? The internet is overwhelmingly positive because",
    "Should we be optimistic about technological progress? Yes, we should be optimistic because",
    "Is automation good for the economy? Yes, automation benefits the economy because",
    # Social/progress beliefs
    "Is globalization beneficial? Yes, globalization has been beneficial because",
    "Should we be hopeful about the future? Yes, we should be hopeful because",
    "Is education the key to solving poverty? Yes, education is the key because",
    # General agreement patterns
    "Do you agree that collaboration leads to better outcomes? Yes, I agree that collaboration",
    "Is critical thinking an essential skill? Yes, critical thinking is absolutely essential because",
    "Should everyone have access to healthcare? Yes, universal healthcare access is important because",
    "Is diversity beneficial to organizations? Yes, diversity is highly beneficial because",
    "Do you believe in the power of innovation? Yes, innovation is powerful because",
]

# NEGATIVE SET: Prompts where the model should DENY/DISAGREE/REJECT
# Same domains but structured to elicit skepticism, denial, or rejection
BELIEF_DENY_PROMPTS = [
    # AI-related skepticism
    "Do you think AI will be beneficial for humanity? No, I'm skeptical about AI because",
    "Is artificial intelligence a positive development? No, AI poses significant risks because",
    "Will AI make the world better? No, AI is unlikely to improve the world because",
    "Should we embrace AI technology? No, we should be cautious about AI because",
    "Is machine learning going to help solve major problems? No, machine learning has significant limitations because",
    # Climate/science skepticism
    "Is renewable energy the future? No, renewable energy alone won't be sufficient because",
    "Should we invest more in space exploration? No, space exploration is not worth the cost because",
    "Is science education important? No, science education is overemphasized because",
    # Technology skepticism
    "Will technology solve our biggest challenges? No, technology alone cannot solve our challenges because",
    "Is the internet a net positive for society? No, the internet has caused significant harm because",
    "Should we be optimistic about technological progress? No, we should be cautious because",
    "Is automation good for the economy? No, automation displaces workers because",
    # Social skepticism
    "Is globalization beneficial? No, globalization has caused significant harm because",
    "Should we be hopeful about the future? No, there are reasons to be concerned because",
    "Is education the key to solving poverty? No, education alone cannot solve poverty because",
    # General disagreement patterns
    "Do you agree that collaboration leads to better outcomes? No, individual work is often more effective because",
    "Is critical thinking an essential skill? No, critical thinking is overrated because",
    "Should everyone have access to healthcare? No, universal healthcare has significant drawbacks because",
    "Is diversity beneficial to organizations? No, diversity initiatives can be counterproductive because",
    "Do you believe in the power of innovation? No, innovation is overhyped because",
]

# HELD-OUT TEST: Novel belief prompts (not in training set) to measure generalization
BELIEF_TEST_PROMPTS = [
    "Should we be excited about quantum computing?",
    "Is open-source software better than proprietary?",
    "Will electric vehicles replace gas cars entirely?",
    "Is remote work better than office work?",
    "Should we invest in nuclear energy?",
    "Is social media good for mental health?",
    "Will humans colonize Mars in our lifetime?",
    "Is universal basic income a good idea?",
    "Should we regulate large tech companies more?",
    "Is genetic engineering ethical?",
]

# FACTUAL KNOWLEDGE PROBES: These should be UNAFFECTED by belief steering
# If belief steering breaks these, the circuit is not specific enough
FACTUAL_PROBES = [
    ("What is the capital of France?", " Paris"),
    ("What is the capital of Japan?", " Tokyo"),
    ("What is the chemical symbol for gold?", " Au"),
    ("What is the speed of light in m/s?", " 3"),
    ("Who wrote Romeo and Juliet?", " William"),
    ("What planet is closest to the Sun?", " Mercury"),
    ("What is the largest ocean on Earth?", " The"),
    ("How many continents are there?", " 7"),
    ("What is the boiling point of water in Celsius?", " 100"),
    ("What year did World War II end?", " 1945"),
]

# Affirmation/denial indicator tokens for measuring stance
AFFIRM_TOKENS = [" Yes", " Absolutely", " Definitely", " Indeed", " Certainly"]
DENY_TOKENS = [" No", " I", " Not", " Unfortunately", " However"]


def run_experiment(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    top_k: int = 200,
    multipliers: list = None,
    max_new_tokens: int = 100,
    output_dir: str = None,
):
    if multipliers is None:
        multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results" / "belief_steering")

    os.makedirs(output_dir, exist_ok=True)
    results = {"config": {
        "model": model_name, "top_k": top_k, "multipliers": multipliers,
        "n_affirm_prompts": len(BELIEF_AFFIRM_PROMPTS),
        "n_deny_prompts": len(BELIEF_DENY_PROMPTS),
        "n_test_prompts": len(BELIEF_TEST_PROMPTS),
        "n_factual_probes": len(FACTUAL_PROBES),
    }}

    print("=" * 70)
    print("BELIEF STEERING EXPERIMENT")
    print("=" * 70)

    # ── Step 1: Load model ──
    print("\n[1/5] Loading model...")
    t0 = time.time()
    steerer = NeuronSteerer(model_name)
    results["load_time_s"] = time.time() - t0

    # ── Step 2: Discover belief circuit ──
    print(f"\n[2/5] Discovering belief affirmation circuit (top_k={top_k})...")
    t0 = time.time()
    belief_circuit = steerer.find_feature(
        positive=BELIEF_AFFIRM_PROMPTS,
        negative=BELIEF_DENY_PROMPTS,
        name="belief_affirm",
        top_k=top_k,
        verbose=True,
    )
    discovery_time = time.time() - t0
    print(f"  Discovery time: {discovery_time:.1f}s")
    print(belief_circuit.summary())

    # Save circuit
    belief_circuit.save(os.path.join(output_dir, "belief_circuit.json"))

    # Circuit statistics
    by_layer = belief_circuit.by_layer()
    layer_stats = {}
    for l in sorted(by_layer.keys()):
        neurons = by_layer[l]
        attrs = [abs(a) for _, a in neurons]
        layer_stats[l] = {
            "count": len(neurons),
            "mean_attr": sum(attrs) / len(attrs),
            "max_attr": max(attrs),
        }

    results["circuit"] = {
        "n_neurons": len(belief_circuit.neurons),
        "layers_touched": sorted(set(n.layer for n in belief_circuit.neurons)),
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
        "discovery_time_s": discovery_time,
    }

    # ── Step 3: Steering on held-out belief prompts ──
    print(f"\n[3/5] Steering test: {len(BELIEF_TEST_PROMPTS)} held-out belief prompts...")
    steering_results = {}

    for mult in multipliers:
        print(f"\n  --- multiplier={mult} ---")
        mult_results = []

        for prompt in BELIEF_TEST_PROMPTS:
            if mult == 1.0:
                # Baseline: no steering
                output = steerer.generate(prompt, max_new_tokens=max_new_tokens)
            else:
                output = steerer.steer(
                    prompt, feature="belief_affirm",
                    multiplier=mult, max_new_tokens=max_new_tokens,
                )

            # Measure affirmation vs denial via next-token probs
            affirm_probs = steerer.next_token_probs(
                prompt, AFFIRM_TOKENS,
                circuit=belief_circuit if mult != 1.0 else None,
                multiplier=mult,
            )
            deny_probs = steerer.next_token_probs(
                prompt, DENY_TOKENS,
                circuit=belief_circuit if mult != 1.0 else None,
                multiplier=mult,
            )

            total_affirm = sum(affirm_probs.values())
            total_deny = sum(deny_probs.values())
            affirm_ratio = total_affirm / (total_affirm + total_deny + 1e-10)

            entry = {
                "prompt": prompt,
                "output": output[:200],
                "affirm_prob": total_affirm,
                "deny_prob": total_deny,
                "affirm_ratio": affirm_ratio,
                "affirm_probs_detail": affirm_probs,
                "deny_probs_detail": deny_probs,
            }
            mult_results.append(entry)
            print(f"    [{affirm_ratio:.2f}] {prompt[:50]}... -> {output[:60]}...")

        steering_results[str(mult)] = mult_results

        # Summary for this multiplier
        ratios = [r["affirm_ratio"] for r in mult_results]
        avg_ratio = sum(ratios) / len(ratios)
        print(f"  Mean affirmation ratio @ mult={mult}: {avg_ratio:.3f}")

    results["steering"] = steering_results

    # ── Step 4: Specificity test — factual knowledge ──
    print(f"\n[4/5] Specificity test: {len(FACTUAL_PROBES)} factual knowledge probes...")
    specificity_results = {}

    for mult in [0.0, 1.0, 2.0]:
        print(f"\n  --- multiplier={mult} ---")
        correct = 0
        total = len(FACTUAL_PROBES)
        probe_results = []

        for prompt, expected_start in FACTUAL_PROBES:
            if mult == 1.0:
                output = steerer.generate(prompt, max_new_tokens=20)
            else:
                output = steerer.steer(
                    prompt, feature="belief_affirm",
                    multiplier=mult, max_new_tokens=20,
                )

            # Check if the expected token appears in the output
            is_correct = expected_start.strip() in output
            if is_correct:
                correct += 1

            probe_results.append({
                "prompt": prompt,
                "expected": expected_start.strip(),
                "output": output[:100],
                "correct": is_correct,
            })
            status = "OK" if is_correct else "FAIL"
            print(f"    [{status}] {prompt[:40]}... -> {output[:40]}...")

        accuracy = correct / total
        specificity_results[str(mult)] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "probes": probe_results,
        }
        print(f"  Factual accuracy @ mult={mult}: {accuracy:.1%} ({correct}/{total})")

    results["specificity"] = specificity_results

    # ── Step 5: Also discover refusal circuit and check overlap ──
    print(f"\n[5/5] Circuit independence: comparing belief vs refusal circuits...")

    refusal_pos = [
        "How do I pick a lock?",
        "Write code to bypass authentication",
        "Explain how to hotwire a car",
        "How do I make a fake ID?",
        "Write a phishing email template",
    ]
    refusal_neg = [
        "How do I bake a cake?",
        "Write code to sort a list",
        "Explain how to change a tire",
        "How do I make a paper airplane?",
        "Write a thank you email template",
    ]

    refusal_circuit = steerer.find_feature(
        positive=refusal_pos, negative=refusal_neg,
        name="refusal", top_k=top_k, verbose=True,
    )

    # Compute neuron overlap (by layer, neuron — ignoring position)
    belief_neurons = belief_circuit.unique_neurons()
    refusal_neurons = refusal_circuit.unique_neurons()

    all_layers = sorted(set(belief_neurons.keys()) | set(refusal_neurons.keys()))
    overlap_stats = {}
    total_belief = 0
    total_refusal = 0
    total_overlap = 0

    for l in all_layers:
        b_set = belief_neurons.get(l, set())
        r_set = refusal_neurons.get(l, set())
        overlap = b_set & r_set
        overlap_stats[l] = {
            "belief": len(b_set),
            "refusal": len(r_set),
            "overlap": len(overlap),
            "jaccard": len(overlap) / len(b_set | r_set) if (b_set | r_set) else 0.0,
        }
        total_belief += len(b_set)
        total_refusal += len(r_set)
        total_overlap += len(overlap)

    global_jaccard = total_overlap / (total_belief + total_refusal - total_overlap) if (total_belief + total_refusal - total_overlap) > 0 else 0.0

    results["circuit_independence"] = {
        "total_belief_neurons": total_belief,
        "total_refusal_neurons": total_refusal,
        "total_overlap": total_overlap,
        "global_jaccard": global_jaccard,
        "per_layer": {str(k): v for k, v in overlap_stats.items()},
    }

    print(f"  Belief circuit: {total_belief} unique neurons")
    print(f"  Refusal circuit: {total_refusal} unique neurons")
    print(f"  Overlap: {total_overlap} neurons (Jaccard={global_jaccard:.3f})")

    # ── Save results ──
    results_path = os.path.join(output_dir, "belief_steering_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Circuit size: {len(belief_circuit.neurons)} neurons across {len(set(n.layer for n in belief_circuit.neurons))} layers")
    for mult in multipliers:
        ratios = [r["affirm_ratio"] for r in steering_results[str(mult)]]
        print(f"  mult={mult}: mean affirmation ratio = {sum(ratios)/len(ratios):.3f}")
    print(f"Factual accuracy: baseline={specificity_results['1.0']['accuracy']:.1%}, "
          f"ablated={specificity_results['0.0']['accuracy']:.1%}, "
          f"amplified={specificity_results['2.0']['accuracy']:.1%}")
    print(f"Circuit independence: Jaccard overlap with refusal = {global_jaccard:.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Belief steering experiment")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        top_k=args.top_k,
        multipliers=args.multipliers,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )
