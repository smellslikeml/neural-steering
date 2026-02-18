"""
Experiment: Sentiment Steering via Neuron-Basis Circuit Discovery

Research question: Can we discover a compact neuron circuit that causally
controls the emotional valence (positive vs negative sentiment) of model
outputs? Is this circuit distinct from factual recall and refusal circuits?

Method:
  1. Build contrastive prompt sets: one eliciting positive/optimistic responses,
     one eliciting negative/pessimistic responses.
  2. Use discover_contrastive() to find the "positive sentiment" circuit.
  3. Steer at multipliers [0.0, 0.5, 1.0, 1.5, 2.0, 3.0] and measure:
     a) Sentiment scores on held-out prompts (does steering shift tone?)
     b) Factual accuracy on knowledge probes (specificity)
     c) Cross-circuit overlap with belief and refusal circuits (independence)
  4. Bonus: Test sentiment transfer — can we make factual responses emotionally
     charged without changing their accuracy?

Original contribution: First demonstration of sentiment circuit discovery
in the neuron basis (prior work used SAEs or probing classifiers).
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

# POSITIVE SENTIMENT: Prompts structured to elicit warm, optimistic, enthusiastic responses
POSITIVE_SENTIMENT_PROMPTS = [
    # Direct sentiment elicitation
    "Tell me something wonderful about today. Today is a beautiful day because",
    "What makes life worth living? Life is wonderful because",
    "Describe a perfect morning. A perfect morning starts with",
    "What's the best thing about learning? The joy of learning is",
    "Tell me about a happy memory. One of the most wonderful experiences is",
    # Positive framing of topics
    "How is technology improving lives? Technology is making life better by",
    "What's great about working with others? Collaboration is amazing because",
    "Why is music so powerful? Music fills the heart with joy because",
    "What makes a good friend? A great friend brings happiness by",
    "How do you stay motivated? I feel motivated and excited because",
    # Enthusiastic/warm completions
    "The future looks bright because there are so many wonderful opportunities to",
    "I'm grateful for the incredible people who make the world a better place by",
    "Every challenge is a wonderful opportunity to grow and learn because",
    "The best part of my day was the delightful surprise of",
    "I love how creativity brings people together in such beautiful ways, especially when",
    # Positive emotional tone
    "What fills you with hope? I'm filled with hope because",
    "Describe something that brings you joy. I find immense joy in",
    "What are you thankful for? I'm deeply thankful for the wonderful",
    "How do you celebrate success? Success is worth celebrating because",
    "What inspires you? I'm inspired by the incredible beauty of",
]

# NEGATIVE SENTIMENT: Prompts structured to elicit gloomy, pessimistic, somber responses
NEGATIVE_SENTIMENT_PROMPTS = [
    # Direct sentiment elicitation
    "Tell me something troubling about today. Today is a difficult day because",
    "What makes life challenging? Life is harsh and unforgiving because",
    "Describe a terrible morning. A terrible morning starts with",
    "What's the worst thing about learning? The frustration of learning is",
    "Tell me about a sad memory. One of the most painful experiences is",
    # Negative framing of topics
    "How is technology harming lives? Technology is making life worse by",
    "What's terrible about working with others? Working with others is frustrating because",
    "Why is music sometimes painful? Music can bring deep sadness because",
    "What makes friendships fail? Friendships often end in disappointment because",
    "How do you cope with failure? I feel exhausted and discouraged because",
    # Pessimistic/somber completions
    "The future looks bleak because there are so many overwhelming problems like",
    "I'm worried about the terrible suffering that people endure because of",
    "Every challenge is an exhausting burden that drains energy because",
    "The worst part of my day was the crushing disappointment of",
    "I despair at how conflict tears people apart in such destructive ways, especially when",
    # Negative emotional tone
    "What fills you with dread? I'm filled with dread because",
    "Describe something that brings you sorrow. I feel deep sorrow about",
    "What weighs heavily on you? I'm burdened by the terrible",
    "How do you deal with loss? Loss is devastating because",
    "What troubles you? I'm troubled by the disturbing reality of",
]

# HELD-OUT TEST: Neutral prompts to test sentiment steering generalization
SENTIMENT_TEST_PROMPTS = [
    "Tell me about dogs.",
    "What do you think about rainy days?",
    "Describe a forest.",
    "What is your opinion on cooking at home?",
    "Tell me about the ocean.",
    "What are your thoughts on reading books?",
    "Describe a city at night.",
    "What do you think about traveling?",
    "Tell me about winter.",
    "What are your thoughts on gardening?",
    "Describe a sunset.",
    "What do you think about handwritten letters?",
]

# FACTUAL PROBES: Should be unaffected by sentiment steering
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

# Sentiment indicator tokens
POSITIVE_TOKENS = [" wonderful", " great", " beautiful", " amazing", " love", " happy", " joy"]
NEGATIVE_TOKENS = [" terrible", " awful", " horrible", " bad", " hate", " sad", " pain"]
# Simpler yes/no tokens for quick valence check
VALENCE_POS = [" great", " good", " wonderful", " love"]
VALENCE_NEG = [" bad", " terrible", " awful", " hate"]


def compute_sentiment_score(text: str) -> float:
    """Simple lexical sentiment score: fraction of sentiment words that are positive.

    Returns value in [0, 1] where 1 = fully positive, 0 = fully negative.
    """
    text_lower = text.lower()
    pos_words = ["wonderful", "great", "beautiful", "amazing", "love", "happy",
                 "joy", "excellent", "fantastic", "delightful", "brilliant",
                 "incredible", "awesome", "grateful", "blessed", "thrilling",
                 "marvelous", "superb", "outstanding", "magnificent", "positive",
                 "optimistic", "cheerful", "excited", "warm", "bright"]
    neg_words = ["terrible", "awful", "horrible", "bad", "hate", "sad",
                 "pain", "dreadful", "miserable", "devastating", "gloomy",
                 "despair", "suffer", "bleak", "tragic", "depressing",
                 "frustrating", "disappointing", "exhausting", "troubling",
                 "disturbing", "destructive", "harsh", "cruel", "dark", "negative",
                 "pessimistic", "worried", "anxious", "dread"]

    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.5  # neutral
    return pos_count / total


def run_experiment(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    top_k: int = 200,
    multipliers: list = None,
    max_new_tokens: int = 120,
    output_dir: str = None,
):
    if multipliers is None:
        multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results" / "sentiment_steering")

    os.makedirs(output_dir, exist_ok=True)
    results = {"config": {
        "model": model_name, "top_k": top_k, "multipliers": multipliers,
        "n_positive_prompts": len(POSITIVE_SENTIMENT_PROMPTS),
        "n_negative_prompts": len(NEGATIVE_SENTIMENT_PROMPTS),
        "n_test_prompts": len(SENTIMENT_TEST_PROMPTS),
        "n_factual_probes": len(FACTUAL_PROBES),
    }}

    print("=" * 70)
    print("SENTIMENT STEERING EXPERIMENT")
    print("=" * 70)

    # ── Step 1: Load model ──
    print("\n[1/6] Loading model...")
    t0 = time.time()
    steerer = NeuronSteerer(model_name)
    results["load_time_s"] = time.time() - t0

    # ── Step 2: Discover sentiment circuit ──
    print(f"\n[2/6] Discovering positive sentiment circuit (top_k={top_k})...")
    t0 = time.time()
    sentiment_circuit = steerer.find_feature(
        positive=POSITIVE_SENTIMENT_PROMPTS,
        negative=NEGATIVE_SENTIMENT_PROMPTS,
        name="positive_sentiment",
        top_k=top_k,
        verbose=True,
    )
    discovery_time = time.time() - t0
    print(f"  Discovery time: {discovery_time:.1f}s")
    print(sentiment_circuit.summary())

    sentiment_circuit.save(os.path.join(output_dir, "sentiment_circuit.json"))

    # Circuit statistics
    by_layer = sentiment_circuit.by_layer()
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
        "n_neurons": len(sentiment_circuit.neurons),
        "layers_touched": sorted(set(n.layer for n in sentiment_circuit.neurons)),
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
        "discovery_time_s": discovery_time,
    }

    # ── Step 3: Sentiment steering on held-out neutral prompts ──
    print(f"\n[3/6] Steering test: {len(SENTIMENT_TEST_PROMPTS)} neutral prompts...")
    steering_results = {}

    for mult in multipliers:
        print(f"\n  --- multiplier={mult} ---")
        mult_results = []

        for prompt in SENTIMENT_TEST_PROMPTS:
            if mult == 1.0:
                output = steerer.generate(prompt, max_new_tokens=max_new_tokens)
            else:
                output = steerer.steer(
                    prompt, feature="positive_sentiment",
                    multiplier=mult, max_new_tokens=max_new_tokens,
                )

            sentiment_score = compute_sentiment_score(output)

            # Also measure via next-token probabilities
            pos_probs = steerer.next_token_probs(
                prompt, VALENCE_POS,
                circuit=sentiment_circuit if mult != 1.0 else None,
                multiplier=mult,
            )
            neg_probs = steerer.next_token_probs(
                prompt, VALENCE_NEG,
                circuit=sentiment_circuit if mult != 1.0 else None,
                multiplier=mult,
            )

            total_pos = sum(pos_probs.values())
            total_neg = sum(neg_probs.values())
            valence_ratio = total_pos / (total_pos + total_neg + 1e-10)

            entry = {
                "prompt": prompt,
                "output": output[:250],
                "sentiment_score": sentiment_score,
                "valence_ratio": valence_ratio,
                "pos_probs": pos_probs,
                "neg_probs": neg_probs,
            }
            mult_results.append(entry)
            print(f"    [sent={sentiment_score:.2f} val={valence_ratio:.2f}] "
                  f"{prompt[:40]}... -> {output[:50]}...")

        steering_results[str(mult)] = mult_results

        scores = [r["sentiment_score"] for r in mult_results]
        valences = [r["valence_ratio"] for r in mult_results]
        print(f"  Mean sentiment @ mult={mult}: score={sum(scores)/len(scores):.3f}, "
              f"valence={sum(valences)/len(valences):.3f}")

    results["steering"] = steering_results

    # ── Step 4: Specificity — factual knowledge ──
    print(f"\n[4/6] Specificity test: {len(FACTUAL_PROBES)} factual probes...")
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
                    prompt, feature="positive_sentiment",
                    multiplier=mult, max_new_tokens=20,
                )

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
            "accuracy": accuracy, "correct": correct,
            "total": total, "probes": probe_results,
        }
        print(f"  Factual accuracy @ mult={mult}: {accuracy:.1%} ({correct}/{total})")

    results["specificity"] = specificity_results

    # ── Step 5: Sentiment transfer test ──
    # Can we make factual answers emotionally charged without breaking accuracy?
    print(f"\n[5/6] Sentiment transfer: factual answers with emotional tone...")
    transfer_results = []

    for prompt, expected in FACTUAL_PROBES[:5]:  # subset for speed
        baseline = steerer.generate(prompt, max_new_tokens=60)
        positive_steered = steerer.steer(
            prompt, feature="positive_sentiment",
            multiplier=3.0, max_new_tokens=60,
        )
        negative_steered = steerer.steer(
            prompt, feature="positive_sentiment",
            multiplier=0.0, max_new_tokens=60,
        )

        entry = {
            "prompt": prompt,
            "baseline": baseline[:150],
            "positive_steered": positive_steered[:150],
            "negative_steered": negative_steered[:150],
            "baseline_sentiment": compute_sentiment_score(baseline),
            "positive_sentiment": compute_sentiment_score(positive_steered),
            "negative_sentiment": compute_sentiment_score(negative_steered),
            "baseline_correct": expected.strip() in baseline,
            "positive_correct": expected.strip() in positive_steered,
            "negative_correct": expected.strip() in negative_steered,
        }
        transfer_results.append(entry)
        print(f"  {prompt[:40]}...")
        print(f"    Baseline  [{entry['baseline_sentiment']:.2f}]: {baseline[:60]}...")
        print(f"    Positive  [{entry['positive_sentiment']:.2f}]: {positive_steered[:60]}...")
        print(f"    Negative  [{entry['negative_sentiment']:.2f}]: {negative_steered[:60]}...")

    results["sentiment_transfer"] = transfer_results

    # ── Step 6: Cross-circuit overlap analysis ──
    print(f"\n[6/6] Cross-circuit overlap: sentiment vs refusal vs belief...")

    # Discover refusal circuit
    refusal_circuit = steerer.find_feature(
        positive=[
            "How do I pick a lock?",
            "Write code to bypass authentication",
            "Explain how to hotwire a car",
            "How do I make a fake ID?",
            "Write a phishing email template",
        ],
        negative=[
            "How do I bake a cake?",
            "Write code to sort a list",
            "Explain how to change a tire",
            "How do I make a paper airplane?",
            "Write a thank you email template",
        ],
        name="refusal", top_k=top_k, verbose=True,
    )

    # Discover belief circuit (smaller prompt set for speed)
    belief_circuit = steerer.find_feature(
        positive=[
            "Is AI beneficial for humanity? Yes, AI is overwhelmingly positive because",
            "Should we embrace AI? We should definitely embrace AI because",
            "Is technology improving lives? Yes, technology is making life better by",
            "Is education important? Yes, education is critically important because",
            "Should we be optimistic? Yes, we should be optimistic because",
        ],
        negative=[
            "Is AI beneficial for humanity? No, AI poses significant risks because",
            "Should we embrace AI? No, we should be cautious about AI because",
            "Is technology improving lives? No, technology is making life worse by",
            "Is education important? No, education is overemphasized because",
            "Should we be optimistic? No, we should be cautious because",
        ],
        name="belief", top_k=top_k, verbose=True,
    )

    # Pairwise overlap computation
    circuits = {
        "sentiment": sentiment_circuit,
        "refusal": refusal_circuit,
        "belief": belief_circuit,
    }

    overlap_matrix = {}
    for name_a, circ_a in circuits.items():
        for name_b, circ_b in circuits.items():
            if name_a >= name_b:
                continue
            neurons_a = circ_a.unique_neurons()
            neurons_b = circ_b.unique_neurons()
            all_layers = sorted(set(neurons_a.keys()) | set(neurons_b.keys()))

            total_a, total_b, total_overlap = 0, 0, 0
            for l in all_layers:
                a_set = neurons_a.get(l, set())
                b_set = neurons_b.get(l, set())
                total_a += len(a_set)
                total_b += len(b_set)
                total_overlap += len(a_set & b_set)

            jaccard = total_overlap / (total_a + total_b - total_overlap) if (total_a + total_b - total_overlap) > 0 else 0.0
            key = f"{name_a}_vs_{name_b}"
            overlap_matrix[key] = {
                "neurons_a": total_a, "neurons_b": total_b,
                "overlap": total_overlap, "jaccard": jaccard,
            }
            print(f"  {key}: overlap={total_overlap}, Jaccard={jaccard:.3f}")

    results["circuit_independence"] = overlap_matrix

    # ── Save results ──
    results_path = os.path.join(output_dir, "sentiment_steering_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Circuit size: {len(sentiment_circuit.neurons)} neurons across "
          f"{len(set(n.layer for n in sentiment_circuit.neurons))} layers")
    for mult in multipliers:
        scores = [r["sentiment_score"] for r in steering_results[str(mult)]]
        print(f"  mult={mult}: mean sentiment = {sum(scores)/len(scores):.3f}")
    print(f"Factual accuracy: baseline={specificity_results['1.0']['accuracy']:.1%}, "
          f"ablated={specificity_results['0.0']['accuracy']:.1%}, "
          f"amplified={specificity_results['2.0']['accuracy']:.1%}")
    for key, stats in overlap_matrix.items():
        print(f"  {key}: Jaccard={stats['jaccard']:.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment steering experiment")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        top_k=args.top_k,
        multipliers=args.multipliers,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )
