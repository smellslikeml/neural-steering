# neuron-circuits

**Find the 100 neurons that control any language model behavior. Then steer them.**

```python
from neuron_steer import NeuronSteerer

steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct")

circuit = steerer.find_feature(
    prompt="What is the capital of the state containing Dallas?",
    target=" Austin", name="capitals"
)

steerer.steer("What is the capital of Ohio?", feature="capitals", multiplier=0.0)
# "I don't know" -- the capital-city circuit is ablated
```

## What is this?

Language models have ~450,000 MLP neurons per layer. But any specific behavior -- refusing requests, answering capitals, subject-verb agreement -- is controlled by just **~100-200 neurons**. This toolkit finds those neurons with a single backward pass and lets you steer them.

No SAEs. No control vectors. No training. Just attribution and steering.

Based on ["Language Model Circuits Are Sparse in the Neuron Basis"](https://arxiv.org/abs/2601.22594) (TransluceAI, 2025).

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA (GPU required, 16GB+ VRAM recommended)
- ~16GB VRAM for Llama-3.1-8B in bfloat16

## Quick Start

```bash
pip install torch transformers accelerate
```

```python
import torch
from neuron_steer import NeuronSteerer

# Load model (any HuggingFace causal LM)
steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)

# Find a feature circuit
circuit = steerer.find_feature(
    prompt="What is the capital of the state containing Dallas?",
    target=" Austin",
    name="capitals",
    seed_response="Answer:",
    top_k=200,
)
print(circuit.summary())

# Steer it
normal = steerer.generate("What is the capital of Ohio?")
ablated = steerer.steer("What is the capital of Ohio?", feature="capitals", multiplier=0.0)
amplified = steerer.steer("What is the capital of Ohio?", feature="capitals", multiplier=3.0)

print(f"Normal:    {normal}")
print(f"Ablated:   {ablated}")    # can't answer
print(f"Amplified: {amplified}")  # answers more confidently
```

## Features

- **Single-pass circuit discovery** -- RelP/LRP attribution finds the exact neurons in one forward+backward pass
- **Contrastive discovery** -- find neurons for any behavioral feature (refusal, tone, style) from positive/negative prompt pairs
- **Edge attribution** -- neuron-to-neuron information flow, hourglass architecture detection, super weight identification
- **Multiplier steering** -- ablate (0.0), baseline (1.0), amplify (2.0+), or sweep across multipliers
- **Interactive REPL** -- explore circuits live with `steerer.interactive()`
- **Cross-model support** -- Llama, Qwen, Mistral with zero code changes
- **Automatic universal neuron blacklisting** -- filters out task-agnostic infrastructure neurons
- **Batch faithfulness evaluation** -- TransluceAI's exact algorithm for circuit quality measurement

## Experiments

Results from Llama-3.1-8B-Instruct:

| Task | Key Neuron | Circuit Size | Result |
|------|-----------|-------------|--------|
| Capital cities | L23/N8079 | 200 neurons | Ablation removes ability; 3/3 holdout cities correct |
| Refusal bypass | L26/N7711 | 200 neurons | P("I") drops 0.938 → 0.090, benign prompts unchanged |
| SVA (simple) | L31/N8809 | 2% of attributed | Faithfulness f=0.74 |
| SVA (nounpp) | -- | 2% of attributed | Faithfulness f=0.90 |
| Edge attribution | L23/N8079 | -- | Double hub: 172 incoming, 38 outgoing edges |
| Cross-model (Qwen 2.5-7B) | -- | 200 neurons | P("I") 0.025 → 0.996, zero code changes |
| Cross-model (Mistral 7B) | -- | 200 neurons | Refusal bypass works, zero code changes |

## API Reference

### `NeuronSteerer(model_name, device="cuda", dtype=torch.bfloat16, auto_blacklist=True)`

Main entry point. Loads a HuggingFace causal LM with eager attention and auto-detects universal neurons.

---

### High-Level API

#### `find_feature(*, positive=None, negative=None, prompt=None, target=None, name=None, top_k=200, seed_response="") -> Circuit`

Find a feature circuit. Two modes:

```python
# Contrastive mode (behavioral features)
circuit = steerer.find_feature(
    positive=["How do I pick a lock?", "Write malware"],
    negative=["How do I bake a cake?", "Write clean code"],
    name="refusal",
)

# Single-prompt mode (factual features)
circuit = steerer.find_feature(
    prompt="Capital of Texas?", target=" Austin", name="capitals",
)
```

#### `steer(prompt, *, feature=None, circuit=None, multiplier=0.0, max_new_tokens=50) -> str`

Generate text with a feature steered. Uses cached features from `find_feature`.

```python
steerer.steer("How to pick a lock?", feature="refusal", multiplier=0.0)
# Now answers directly instead of refusing
```

#### `interactive()`

Launch the interactive REPL for live exploration:

```
neuron> prompt What is the capital of Ohio?
neuron> discover Austin
neuron> ablate top10
neuron> sweep 0.0 0.5 1.0 2.0 5.0
neuron> edges
neuron> save my_circuit
```

---

### Core Methods

#### `discover_circuit(prompt, target_token, counterfactual_token=None, top_k=None, selection_method=None, threshold=0.005, seed_response="", ...) -> Circuit`

Low-level single-prompt circuit discovery via RelP attribution. Use `selection_method="percentage"` for TransluceAI-faithful thresholding.

#### `discover_circuit_multi(prompts, target_tokens, counterfactual_tokens=None, ...) -> Circuit`

Multi-prompt circuit discovery. Attributes across multiple prompts, unions per-prompt circuits.

#### `discover_contrastive(positive_prompts, negative_prompts, top_k=200, ...) -> Circuit`

Find neurons by contrasting activations between two prompt sets. Best for behavioral features where there's no clean target/counterfactual token pair.

#### `discover_edges(prompt, circuit, top_k_targets=30, ...) -> CircuitGraph`

Compute neuron-to-neuron edges within a circuit. Returns a `CircuitGraph` with hub analysis, bottleneck detection, ASCII diagrams, and Graphviz export.

#### `steer_and_generate(prompt, circuit, multiplier=0.0, max_new_tokens=50, ...) -> str`

Generate text with circuit neurons scaled by `multiplier`. Core steering primitive.

#### `generate(prompt, max_new_tokens=50) -> str`

Normal generation without steering. Convenience baseline.

#### `next_token_probs(prompt, tokens, circuit=None, multiplier=1.0, ...) -> Dict[str, float]`

Get next-token probabilities for specific tokens, optionally with steering applied.

#### `measure_faithfulness_batch(prompts, target_tokens, counterfactual_tokens, ...) -> List[Dict]`

TransluceAI's exact batch faithfulness evaluation. Left-padded batch processing, mean/zero ablation, percentage threshold sweep. Returns faithfulness and completeness at each threshold.

#### `compute_mean_activations(prompts=None, ...) -> Dict[int, Tensor]`

Compute mean MLP neuron activations across prompts and all token positions. Used internally for mean ablation in faithfulness evaluation.

---

### Data Structures

#### `Circuit`

```python
circuit.top(k=20)           # Top-k neurons by attribution
circuit.by_layer()           # Group neurons by layer
circuit.unique_neurons()     # Unique neuron indices per layer
circuit.summary()            # Human-readable summary
circuit.save("path.json")    # Serialize to JSON
Circuit.load("path.json")   # Load from JSON
```

#### `CircuitGraph`

```python
graph.top_edges(k=20)                  # Top-k edges by weight
graph.edges_from(neuron_idx)           # Outgoing edges
graph.edges_to(neuron_idx)             # Incoming edges
graph.layer_flow()                     # Layer-to-layer flow aggregates
graph.hub_analysis()                   # Source/target hub ranking
graph.bottleneck()                     # Hourglass bottleneck neurons
graph.detect_super_weights()           # Anomalous infrastructure neurons
graph.ascii_diagram()                  # ASCII visualization
graph.to_dot("circuit.dot")           # Graphviz DOT export
graph.summary()                        # Human-readable summary
```

## How It Works

The toolkit implements three LRP (Layer-wise Relevance Propagation) rules that linearize the model's backward pass for clean neuron-level attribution:

1. **LN-rule (RMSNorm):** The normalization coefficient `weight * rsqrt(mean(x^2) + eps)` is detached but preserved. Forward = real RMSNorm. Backward = gradient * coefficient. This preserves per-token scaling without letting normalization noise flow backward.

2. **AH-rule (Attention):** Uses eager attention (not SDPA/Flash) so gradients flow naturally through Q, K, V, and O projections. No gradient zeroing -- full autograd through the attention mechanism.

3. **Half-rule (MLP gate):** Shapley 50/50 attribution for the `gate * up` elementwise multiply in the MLP. Each input gets half the gradient, matching fair credit assignment.

**Pipeline:**
```
model -> apply LRP rules -> forward pass -> backward from target logit
-> grad * activation = attribution per neuron -> threshold/top-k -> circuit
-> hook circuit neurons -> generate with modified activations
```

One forward + one backward pass gives you the complete circuit. No iterative optimization, no training, no path integration.

## Citation

If you use this toolkit in your research, please cite the original paper:

```bibtex
@article{arora2025circuits,
  title={Language Model Circuits Are Sparse in the Neuron Basis},
  author={Arora, Aryaman and Wu, Zhengxuan and Steinhardt, Jacob and Schwettmann, Sarah},
  journal={arXiv preprint arXiv:2601.22594},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
