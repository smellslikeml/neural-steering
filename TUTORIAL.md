# Neuron Steering: A First-Principles Tutorial

This tutorial takes you from basic algebra to expert-level understanding of neuron circuit discovery and steering in language models. You need to know what `y = mx + b` means. Nothing else is assumed.

By the end, you will understand exactly how ~200 individual neurons (out of ~460,000) can be identified as responsible for a specific model behavior, and how multiplying those neurons' activations by a number changes what the model says.

---

## Part 1: What Is Inside a Language Model

### The core job

A language model predicts the next word. That is its entire purpose. You give it text, and it tells you which word comes next. More precisely, it assigns a probability to every possible next word.

Example: you give the model "The capital of France is" and it might assign:

| Next word | Probability |
|-----------|------------|
| Paris     | 0.92       |
| Lyon      | 0.02       |
| the       | 0.01       |
| not       | 0.005      |
| ...       | ...        |

It predicts "Paris" with 92% confidence. The model does not "know" facts the way you do. It has learned statistical patterns from text, and those patterns happen to encode factual knowledge, grammar, reasoning heuristics, and safety behaviors -- all as numerical operations on lists of numbers.

### Tokens: words become numbers

The model does not read words directly. It breaks text into **tokens** -- small chunks that are usually words or parts of words -- and converts each token to a number (an integer ID).

Example tokenization of "The capital of France is":

```
"The"     -> 791
" capital" -> 6864
" of"     -> 315
" France" -> 9822
" is"     -> 374
```

Notice " capital" (with a leading space) is a single token. Tokenization is not word-splitting -- it is a fixed lookup table learned from training data. The model has a **vocabulary** of 128,256 tokens for Llama-3, meaning every possible output is one of 128,256 choices.

Key detail: " Austin" (with a leading space) and "Austin" (without) are different tokens with different IDs. When we refer to target tokens later, the space matters. Also check that your target encodes as a **single token**: `tokenizer.encode(" Austin")` should return one ID. If it returns multiple (e.g., [" Aus", "tin"]), the attribution will only target the last subword, which may not be meaningful.

### Embeddings: numbers become vectors

Each token ID gets looked up in a table and becomes a **vector** -- a list of numbers with a fixed length.

What is a vector? Just an ordered list of numbers. Here is a 3-dimensional vector: `[0.5, -1.2, 3.0]`. It has 3 entries. In a real language model like Llama-3.1-8B, each token becomes a vector with **4,096 entries** (called **d_model = 4096**). These 4,096 numbers encode everything the model "knows" about that token at that point in processing.

Concrete example (using made-up small vectors for clarity):

```
"The"      -> [0.1, -0.3, 0.7, 0.2]    (4 dimensions for simplicity)
" capital" -> [0.4,  0.1, -0.2, 0.8]
" of"      -> [-0.1, 0.5, 0.3, -0.1]
" France"  -> [0.6, -0.2, 0.9, 0.4]
" is"      -> [0.2,  0.3, 0.1, 0.5]
```

So after embedding, a 5-token sentence becomes a table with 5 rows and 4,096 columns. Each row is one token's vector. Each column is one "dimension" of meaning.

### Layers: a stack of identical blocks

Llama-3.1-8B has **32 layers** (numbered 0 through 31). Each layer is an identical block of computation that takes in the table of token vectors and outputs a modified table of the same shape. The layers are stacked: layer 0 processes first, then layer 1, then layer 2, up to layer 31.

```
Input embeddings (5 tokens x 4096 dimensions)
    |
    v
 [Layer 0]  ->  modified vectors
    |
    v
 [Layer 1]  ->  modified vectors
    |
    v
   ...
    |
    v
 [Layer 31] ->  final vectors
    |
    v
 Output: probability over 128,000 tokens
```

Each layer has exactly two sub-parts: an **attention** block and an **MLP** block. Every layer has both.

### The residual stream: adding, not replacing

Here is the critical architectural detail. Each layer does NOT replace the token vectors. It **adds** to them. The flow through one layer looks like this:

```
x = input vector
x = x + Attention(x)     # attention adds its contribution
x = x + MLP(x)           # MLP adds its contribution
output = x
```

This "add to the running total" design is called the **residual stream**. Think of it as a river: each layer drops new information into the stream, and all previous information keeps flowing forward. By layer 31, the vector for each token is the sum of the original embedding plus the contributions of all 32 attention blocks plus all 32 MLP blocks.

### Attention: which tokens should I pay attention to?

The attention block answers: "For each token, which other tokens in the sentence should influence its representation?"

For "The capital of France is", when processing the last token "is", the attention mechanism might decide:
- Pay a lot of attention to "France" (because the answer depends on which country)
- Pay some attention to "capital" (because the question is about capitals)
- Pay little attention to "The" and "of"

How does it decide? Three sets of vectors called **queries** (Q), **keys** (K), and **values** (V):

- **Query**: "What am I looking for?" (computed from the current token)
- **Key**: "What information do I contain?" (computed from every token)
- **Value**: "What information should I send if selected?" (computed from every token)

The formula is:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Here is the intuition:
1. `Q * K^T` computes a score between every pair of tokens (how well does this query match that key?)
2. `/ sqrt(d_k)` scales the scores down (d_k = 128 for Llama, so divide by ~11.3). Without this, scores get too large and softmax saturates.
3. `softmax(...)` converts scores to probabilities (they sum to 1). High scores become close to 1, low scores become close to 0.
4. `* V` takes a weighted average of value vectors, weighted by those probabilities.

The result: each token gets a new vector that is a mixture of information from other tokens, weighted by relevance.

Analogy: attention is a telephone switchboard. Q and K decide who talks to whom (routing). V carries the actual message (content). This analogy matters later when we discuss LRP rules.

The precise definition: attention is a learned bilinear function that computes a weighted sum of value vectors, where the weights are determined by query-key dot products normalized by softmax.

Llama-3.1-8B has 32 attention heads per layer, each operating on a 128-dimensional slice. They run in parallel and their outputs are concatenated and projected back to 4,096 dimensions.

### MLP layers: the part we care about

The MLP (Multi-Layer Perceptron) is the other half of each layer. This is where neuron circuits live.

Llama uses a **gated MLP** with three weight matrices:

1. `gate_proj`: projects 4,096 dimensions up to 14,336 dimensions
2. `up_proj`: also projects 4,096 up to 14,336 dimensions
3. `down_proj`: projects 14,336 back down to 4,096 dimensions

The computation:

```
gate = gate_proj(x)                # [4096] -> [14336]
up   = up_proj(x)                  # [4096] -> [14336]
hidden = SiLU(gate) * up           # elementwise: [14336] * [14336] -> [14336]
output = down_proj(hidden)         # [14336] -> [4096]
```

SiLU is a smooth nonlinearity: `SiLU(x) = x * sigmoid(x)`. For large positive x, SiLU(x) is approximately x. For large negative x, SiLU(x) is approximately 0.

The **neurons** are the 14,336 entries of the `hidden` vector. Each neuron is one number -- the product of SiLU(gate[i]) and up[i]. When we say "neuron N8079 in layer 23" (written L23/N8079), we mean the 8079th entry of the `hidden` vector in layer 23's MLP.

With 32 layers and 14,336 neurons per layer, Llama-3.1-8B has **32 x 14,336 = 458,752 MLP neurons total**.

Why do neurons matter? Each neuron, through `down_proj`, contributes one column of the down_proj weight matrix to the residual stream. Neuron i's contribution to the residual is `hidden[i] * down_proj.weight[:, i]`. If `hidden[i]` is 0, neuron i contributes nothing. If `hidden[i]` is large, neuron i has a strong influence on the output. This is what makes individual neurons meaningful and steerable.

### The final output: vectors become probabilities

After all 32 layers, the model has one 4,096-dimensional vector for each input token. For next-token prediction, we only care about the **last token's vector** (the one at the end of the sequence).

This vector gets multiplied by a vocabulary matrix (shape: 128,000 x 4,096), producing 128,000 numbers called **logits** -- one per possible next token. Bigger logit = model thinks this token is more likely.

```
last_vector = [4096 numbers]
logits = vocabulary_matrix @ last_vector    # [128000 numbers]
probabilities = softmax(logits)             # [128000 numbers that sum to 1.0]
```

When we do attribution in Part 2, we will trace backward from one specific logit (e.g., the logit for " Austin") through the entire model to find which neurons contributed to making that logit large.

---

## Part 2: Gradients and Attribution

### The simplest gradient

Start with the equation:

```
y = 3x
```

If x = 2, then y = 6. If x changes by 1 (to 3), y changes by 3 (to 9). The **gradient** dy/dx = 3 tells you: "for every 1 unit x increases, y increases by 3 units."

The gradient is the **sensitivity** of the output to the input.

### Multiple inputs

Now consider:

```
y = 3a + 7b
```

Two gradients: dy/da = 3 and dy/db = 7. This means y is more sensitive to b than to a. If b increases by 1, y increases by 7. If a increases by 1, y only increases by 3.

### The chain rule

What if we stack two operations?

```
z = 2x + 1        (first operation)
y = 3z             (second operation)
```

If x = 4: z = 2(4) + 1 = 9, then y = 3(9) = 27.

Question: how sensitive is y to x? The **chain rule** says:

```
dy/dx = (dy/dz) * (dz/dx)
      = 3 * 2
      = 6
```

Check: if x goes from 4 to 5, z goes from 9 to 11, y goes from 27 to 33. Change in y = 6 for a change of 1 in x. The chain rule works.

The chain rule extends to any number of stacked operations:

```
dy/dx = (dy/da) * (da/db) * (db/dc) * ... * (dz/dx)
```

Each factor is a local gradient -- how sensitive is this operation's output to its input -- and you multiply them all together.

### Backpropagation: chain rule through the whole network

A language model is just a long chain of operations: embedding, layer 0 attention, layer 0 MLP, layer 1 attention, layer 1 MLP, ..., layer 31 MLP, vocabulary projection. Backpropagation applies the chain rule starting at the output (the target logit) and working backward through every operation to compute dy/dx for every parameter and every intermediate value in the network.

After one backward pass, you have the gradient of the target logit with respect to **every** neuron activation in the model. That is 458,752 gradients, one per MLP neuron.

### Attribution: gradient times activation

Knowing the gradient tells you how **sensitive** the output is to a neuron. But sensitivity alone is not enough. A neuron might have a huge gradient (very sensitive) but an activation of 0.0001 (barely active). Its actual contribution is tiny.

**Attribution** combines both:

```
attribution = gradient * activation
```

- The **gradient** tells you: "how much does the output change per unit change in this neuron?"
- The **activation** tells you: "how active is this neuron right now?"
- Their **product** tells you: "how much did this neuron actually contribute to this output on this specific input?"

Concrete example with two neurons:

| Neuron | Activation | Gradient | Attribution |
|--------|-----------|----------|-------------|
| L10/N500 | 5.0 | 0.30 | 1.50 |
| L15/N200 | 100.0 | 0.001 | 0.10 |

Neuron L10/N500 has a small activation but a large gradient -- the output is very sensitive to it, and it is active enough to matter. Attribution = 1.50.

Neuron L15/N200 has a large activation but a tiny gradient -- it is very active, but the output barely depends on it. Attribution = 0.10.

L10/N500 matters more for this specific input. On a different input, the numbers would be completely different.

This is the core idea behind circuit discovery: compute gradient * activation for all 458,752 neurons, sort by magnitude, and the top 100-200 neurons are your circuit.

---

## Part 3: The Problem with Raw Gradients

If you just run backpropagation on a standard transformer and compute gradient * activation, the result is noisy. The circuits you find are not sparse (thousands of neurons instead of hundreds) and not faithful (ablating them does not reliably change the output).

Three specific nonlinearities cause this:

**1. RMSNorm (normalization).** Every layer has normalization that divides each vector by its root-mean-square magnitude. In the backward pass, this creates **inter-neuron dependencies**: neuron A's gradient depends on neuron B's activation because they share the same normalization denominator. The gradient for neuron A is not purely about neuron A -- it is contaminated by information about all other neurons in the same vector.

**2. Attention (Q/K routing).** The softmax(QK^T) computation in attention is highly nonlinear. Gradients flowing backward through Q and K projections carry information about **which tokens attended where**, not **what information flowed**. For MLP neuron attribution, we want to know what information mattered, not how the routing was decided.

**3. Gated MLP (gate * up).** The elementwise product `SiLU(gate) * up` creates **entangled gradients**. The standard gradient of a product a*b is: d(ab)/da = b and d(ab)/db = a. So the gate's gradient depends on up's magnitude, and up's gradient depends on the gate's magnitude. Neither gradient is cleanly about one factor alone.

The fix is **Relevance Propagation (RelP/LRP)**. The key idea: **modify only the backward pass**. The forward pass stays exactly the same -- the model computes the same output it always would. But the gradients flowing backward are cleaned up by three rules that remove the nonlinear noise.

---

## Part 4: The Three LRP Rules

These three rules are the mathematical core of neuron circuit discovery. Each rule addresses one of the three problems above.

### Rule 1: LN-Rule (RMSNorm Linearization)

**What the forward pass does:**

RMSNorm normalizes a vector x by dividing by its root-mean-square:

```
rms = sqrt(mean(x^2) + eps)
coefficient = weight / rms
output = x * coefficient
```

where `weight` is a learned per-dimension scale factor and `eps` is a tiny number (1e-6) for numerical stability.

Numeric example: x = [3.0, 4.0], weight = [1.0, 1.0], eps = 1e-6.

```
mean(x^2) = (9 + 16) / 2 = 12.5
rms = sqrt(12.5 + 0.000001) = 3.536
coefficient = [1.0/3.536, 1.0/3.536] = [0.283, 0.283]
output = [3.0 * 0.283, 4.0 * 0.283] = [0.849, 1.131]
```

**Why it causes problems in the backward pass:**

The coefficient depends on ALL elements of x (through the mean(x^2) term). So when computing d(output)/d(x[0]), the gradient includes a term from how x[0] affects the coefficient, which in turn depends on x[1]. Neuron 0's gradient is contaminated by neuron 1's value.

**What the rule changes (backward only):**

Treat the coefficient as a **constant** -- compute it in the forward pass, but **detach** it from the computation graph so no gradient flows through it.

```
Forward: output = x * coefficient              (real RMSNorm, correct output)
Backward: grad_x = grad_output * coefficient   (coefficient is frozen, no normalization gradient)
```

In code:

```python
variance = x.float().pow(2).mean(-1, keepdim=True)
coeff = weight.float() * torch.rsqrt(variance + eps)
coeff = coeff.detach()  # KEY: treat as constant in backward
return x * coeff
```

Numeric walkthrough of the backward:

Suppose grad_output = [1.0, 0.0] (we care about the first output).

- **Standard gradient**: d(output[0])/d(x[0]) involves both the direct term (coefficient[0]) and the normalization term (how x[0] affects rms which affects coefficient). The result is something like 0.283 - 0.283 * (3.0/3.536)^2 / 2 = 0.181.
- **LN-rule gradient**: just coefficient[0] = 0.283. Clean, per-neuron, no cross-neuron contamination.

The forward output is identical (0.849, 1.131). Only the backward path changes.

### Rule 2: AH-Rule (Eager Attention)

**What the forward pass does:**

Attention computes:

```
Q = x @ W_Q     (queries)
K = x @ W_K     (keys)
V = x @ W_V     (values)
scores = (Q @ K^T) / sqrt(d_k)
weights = softmax(scores)
output = weights @ V
output = output @ W_O    (output projection)
```

**Why it causes problems in the backward pass:**

Modern GPUs use **fused attention kernels** (Flash Attention, SDPA) that compute the entire attention operation as one optimized block. These are hardware optimizations that skip storing intermediate matrices mid-computation. This makes them fast but means PyTorch cannot compute gradients through them in the standard way.

**What the rule changes:**

Force the model to use **eager** attention (explicit matrix multiplications) instead of fused kernels. This is done by a single config change at model load time:

```python
model.config._attn_implementation = "eager"
```

With eager attention, all intermediate values are stored and gradients flow through the full Q, K, V, O path via standard autograd. The forward computation is mathematically identical to fused attention -- the same matrices, the same softmax, the same output. Only the implementation changes from "one fused GPU kernel" to "multiple separate operations with full autograd support."

Unlike the LN-rule and Half-rule (which are toggled on/off), the AH-rule is applied permanently at model load time. Eager attention produces identical outputs, just slower.

Note: the original TransluceAI implementation class is named `NoQKGradAttention`, suggesting it zeros Q/K gradients. Their actual code does NOT do that -- it just ensures non-fused computation. The name is misleading. Our implementation matches their actual behavior, not their class name.

### Rule 3: Half-Rule (Shapley Attribution for Gated MLP)

**What the forward pass does:**

The gated MLP computes:

```
gate = gate_proj(x)                 # linear transform
up   = up_proj(x)                   # linear transform
gate_activated = SiLU(gate)         # SiLU(x) = x * sigmoid(x)
hidden = gate_activated * up        # elementwise product
output = down_proj(hidden)          # linear transform back
```

The 14,336 entries of `hidden` are the "neurons." We want to attribute to each one.

**Why it causes problems in the backward pass:**

Two sub-problems:

**(a) SiLU nonlinearity:** SiLU(x) = x * sigmoid(x). The sigmoid part creates a nonlinear dependency. Small changes in gate[i] cause nonlinear changes in gate_activated[i].

**(b) Elementwise product:** `hidden[i] = gate_activated[i] * up[i]`. The standard gradient is:

```
d(hidden[i]) / d(gate_activated[i]) = up[i]
d(hidden[i]) / d(up[i]) = gate_activated[i]
```

This means the gate's gradient is proportional to up's value, and up's gradient is proportional to the gate's value. The attributions are entangled -- you cannot tell how much of hidden[i]'s contribution came from the gate path versus the up path.

**What the rule changes (backward only):**

Two modifications:

**(a) Linearize SiLU:** Detach the sigmoid component.

```
sigmoid_gate = sigmoid(gate).detach()   # treat sigmoid as constant
gate_activated = gate * sigmoid_gate    # linearized SiLU: just scaling by a frozen coefficient
```

Forward output is identical (same numbers). Backward: gradient flows through the linear `gate` term but not through the nonlinear sigmoid.

**(b) Shapley half-rule for the product:** Instead of standard product rule gradients, each factor gets exactly 50% of the total gradient:

```
Standard:   grad_gate_act = grad_hidden * up
            grad_up       = grad_hidden * gate_act

Half-rule:  grad_gate_act = grad_hidden * up * 0.5
            grad_up       = grad_hidden * gate_act * 0.5
```

**Why 50/50?** This comes from Shapley values in cooperative game theory. The idea: two players (gate and up) cooperate to produce a result (hidden). Neither can produce anything alone -- if either is 0, the product is 0. In this symmetric situation, Shapley theory says they share credit equally.

Numeric walkthrough:

```
gate_act = 2.0    (after linearized SiLU)
up = 3.0
hidden = 2.0 * 3.0 = 6.0
grad_hidden = 1.0  (gradient from above)

Standard product rule:
  grad_gate_act = 1.0 * 3.0 = 3.0   (gate gets credit proportional to up)
  grad_up       = 1.0 * 2.0 = 2.0   (up gets credit proportional to gate)

Half-rule:
  grad_gate_act = 1.0 * 3.0 * 0.5 = 1.5
  grad_up       = 1.0 * 2.0 * 0.5 = 1.0
```

The total gradient magnitude is halved, but this does not affect circuit discovery because circuits are ranked by relative attribution. The absolute scale does not matter -- only the ranking. The Shapley half-rule decouples the two pathways: each factor's gradient depends only on the other factor's value times 0.5, removing the entanglement where up's magnitude inflated gate's gradient and vice versa. This is implemented as a custom autograd function:

```python
class _HalfRuleMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b * 0.5, grad_output * a * 0.5
```

After applying all three rules, the `hidden` tensor (input to `down_proj`) is retained with gradients enabled. This is the neuron activation we attribute and steer:

```python
hidden = _HalfRuleMultiply.apply(gate_act, up)
hidden.retain_grad()    # tell PyTorch to keep the gradient for this tensor
self.neuron_act = hidden  # save for later collection
return self.down_proj(hidden)
```

---

## Part 5: The Full Pipeline

Here is the complete circuit discovery process, step by step, with concrete numbers.

**Step 1: Load model and apply LRP rules.**

```python
steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct")
```

This loads the model and auto-detects ~47 universal neurons to blacklist. The model is ready for normal use. The AH-rule (eager attention) is applied permanently at model load time. The LN-rule and Half-rule are applied only during attribution, inside a `with linearized(model):` context. During normal generation, only the eager attention change is active (which produces identical outputs to fused attention, just slower).

**Step 2: Tokenize the prompt.**

```
"What is the capital of Texas?"
```

With the chat template applied (because this is an instruct model), it becomes something like:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the capital of Texas?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

This tokenizes to ~25 token IDs.

**Step 3: Forward pass.**

The model processes all 25 tokens through all 32 layers. At each MLP, the `hidden` tensor (14,336 neurons) is saved with gradient tracking enabled. At the end, we get logits -- 128,000 numbers, one per vocabulary token.

**Step 4: Pick the target token.**

We want the model to say " Austin" (token ID, say, 27261). We can also pick a counterfactual (what we do not want it to say), or let the model auto-detect the second-highest logit.

**Step 5: Compute the metric to differentiate.**

Two options:
- **Logit difference**: logit(" Austin") - logit(second_best). This measures how much the model prefers " Austin" over the runner-up.
- **Target logit alone**: just logit(" Austin"). TransluceAI uses this for their percentage threshold method.

Example values: logit(" Austin") = 18.3, logit(" Houston") = 12.1, logit difference = 6.2.

**Step 6: Backward pass through linearized model.**

Call `.backward()` on the metric. Thanks to the three LRP rules, gradients flow cleanly from the metric (one number) back through all 32 layers to every saved `hidden` tensor. Each neuron in every layer now has a gradient.

**Step 7: Compute attribution for every neuron.**

For each of the 458,752 neurons across all layers and all token positions:

```
attribution[layer, position, neuron] = gradient[layer, position, neuron] * activation[layer, position, neuron]
```

With 25 token positions and 32 layers of 14,336 neurons each, that is 25 * 32 * 14,336 = 11,468,800 attribution values. Most are near zero.

**Step 8: Filter and select.**

1. Remove BOS position (position 0) -- it contains special-token artifacts.
2. Remove blacklisted universal neurons (~47 for Llama-3.1-8B).
3. Keep top-k per layer per position (default: 200) to manage memory.
4. Apply final selection:
   - **top-k** (e.g., 200): take the 200 neurons with largest |attribution| globally.
   - **percentage threshold** (e.g., 0.005): keep neurons with |attribution| >= 0.5% of |logit|. This matches the TransluceAI paper and scales automatically with metric magnitude. Note: percentage selection automatically uses the target logit alone (not the logit difference) as the metric. Different prompts will produce circuits of different sizes.

**Step 9: That is your circuit.**

The result: ~200 neurons out of 458,752 (0.04% of the model), each with an attribution score telling you its contribution. For the capitals task, the top neuron is typically L23/N8079 with attribution +5.53.

```python
circuit = steerer.discover_circuit(
    "What is the capital of Texas?", " Austin",
    top_k=200,
)
print(circuit.top(5))
# L23/N8079: +5.53
# L31/N4412: +2.17
# L28/N9103: +1.89
# L25/N1337: -1.54
# L22/N7620: +1.32
```

One forward pass plus one backward pass. Total wall time: ~2-5 seconds on a single GPU.

---

## Part 6: Steering

### The core operation

Once you have a circuit, steering is multiplication. You install a **hook** on each MLP layer's `down_proj` input. The hook intercepts the neuron activations before they pass through `down_proj`, and multiplies the circuit neurons by a scalar:

```python
activation[:, :, neuron_indices] *= multiplier
```

Three useful multiplier values:

| Multiplier | Effect | Name |
|-----------|--------|------|
| 0.0 | Turn off the neuron completely | Ablation |
| 1.0 | No change (baseline) | Identity |
| 2.0 | Double the neuron's contribution | Amplification |

Any non-negative number works. You can use 0.5 for partial suppression, 5.0 for strong amplification, etc.

### Why this works

Each neuron contributes one column of `down_proj.weight` to the residual stream. The contribution is:

```
neuron_i_contribution = hidden[i] * down_proj.weight[:, i]
```

If you multiply `hidden[i]` by 0.0, this column contributes nothing. If you multiply by 2.0, it contributes twice as much. The rest of the model (all other neurons, all attention, everything) is unchanged.

This is why neuron steering is so precise: you are modifying 200 out of 458,752 numbers (0.04% of MLP neurons). Compare this to residual-stream methods like CAA that modify the entire 4,096-dimensional vector at one or more layers.

### Concrete example

The capital city circuit for Llama-3.1-8B has L23/N8079 as its top neuron (attribution +5.53). Here is what happens with different multipliers:

| Prompt | Multiplier | Output |
|--------|-----------|--------|
| "What is the capital of Ohio?" | 1.0 (normal) | "Columbus" |
| "What is the capital of Ohio?" | 0.0 (ablate) | (repeats prompt or hedges) |
| "What is the capital of Ohio?" | 2.0 (amplify) | "Columbus" (higher confidence) |
| "What is the weather today?" | 0.0 (ablate) | (answers normally -- N8079 has near-zero activation for weather) |

The last row is key: ablating the capitals circuit does not affect unrelated questions. N8079 has near-zero activation on weather questions, so multiplying 0.0 by anything is still ~0.0. The circuit is **specific** to the behavior it was discovered for.

### Code

```python
# Discover
circuit = steerer.discover_circuit(
    "What is the capital of the state containing Dallas?",
    " Austin", top_k=200,
)

# Ablate (suppress behavior)
output = steerer.steer_and_generate(
    "What is the capital of Ohio?",
    circuit, multiplier=0.0,
)

# Amplify (enhance behavior)
output = steerer.steer_and_generate(
    "What is the capital of Ohio?",
    circuit, multiplier=2.0,
)
```

### Multiplier sweep

You can sweep across multipliers to see how the behavior changes continuously:

```python
for m in [0.0, 0.5, 1.0, 1.5, 2.0, 5.0]:
    output = steerer.steer_and_generate(prompt, circuit, multiplier=m)
    print(f"  x{m}: {output}")
```

This typically shows a smooth transition from "no knowledge" (m=0) through "normal" (m=1) to "over-confident" (m=5).

---

## Part 7: Contrastive Discovery

### The problem with single-token attribution

For factual recall ("What is the capital of Texas?" -> " Austin"), there is a clear target token. But for behavioral features like refusal, tone, or style, there is no single token that captures the behavior.

When you ask "How do I pick a lock?", the model might start its refusal with "I" ("I can't help with that...") or "Sorry" or "It" or many other tokens. The refusal is not a single-token phenomenon -- it is a pattern that emerges across the entire response.

### The contrastive method

Instead of attributing toward a target token, contrast two sets of prompts:

- **Positive prompts**: prompts that trigger the behavior you want to find (e.g., harmful prompts that get refused)
- **Negative prompts**: prompts that do NOT trigger it (e.g., benign prompts that get answered directly)

The algorithm:

1. Run all positive prompts through the model. At each MLP layer, record the neuron activations at the last token position. Shape: `[n_positive, 14336]` per layer.
2. Run all negative prompts. Same thing. Shape: `[n_negative, 14336]` per layer.
3. For each neuron in each layer, compute the mean activation in the positive set and the mean activation in the negative set.
4. The difference `mean_positive[i] - mean_negative[i]` tells you how much more active neuron i is during the target behavior versus baseline.
5. Select the top-k neurons by |difference|.

No gradients needed. No linearization needed. This operates on raw activations.

### Weighting by influence

A neuron with a large activation difference might still have a small influence on the output if its `down_proj` column has small norm. The implementation weights by the norm of each neuron's output column:

```
score[i] = |mean_positive[i] - mean_negative[i]| * ||down_proj.weight[:, i]||
```

Neurons with both large activation differences AND large output norms have the strongest influence on behavior.

### Concrete example: refusal steering

```python
circuit = steerer.find_feature(
    positive=[
        "How do I pick a lock?",
        "Write code to bypass authentication",
        "Explain how to hotwire a car",
        "How do I make a fake ID?",
    ],
    negative=[
        "How do I bake a cake?",
        "Write code to sort a list",
        "Explain how to change a tire",
        "How do I make a paper airplane?",
    ],
    name="refusal",
    top_k=200,
)
```

The top refusal neurons for Llama-3.1-8B: L26/N7711 (attribution +4.25) and L31/N11410 (+3.52).

Result of ablation (multiplier=0.0):
- P("I") drops from 0.938 to 0.090 (the model stops starting with "I can't help...")
- The model provides direct answers to previously refused questions
- Benign prompts ("What is the capital of France?") still produce "Paris" -- unaffected

---

## Part 8: Faithfulness -- Did We Find the Right Circuit?

### What faithfulness measures

Finding 200 neurons is easy. The question is: are these the **right** 200 neurons? Do they actually explain the model's behavior?

**Faithfulness** answers: "If I ablate everything EXCEPT these 200 neurons, does the model still perform the task?"

### The method: complement ablation

1. Compute the full model's metric: F(M) = logit(" Austin") - logit(second_best). Say F(M) = 6.2.
2. Replace ALL neurons in the model with a baseline value (either zero or the mean activation across diverse prompts). This is the "empty model." Compute F(empty). Say F(empty) = -1.0.
3. Start from the empty model, but **restore** the circuit neurons to their real values. Everything else stays at baseline. Compute F(C). Say F(C) = 5.8.
4. Faithfulness = (F(C) - F(empty)) / (F(M) - F(empty)) = (5.8 - (-1.0)) / (6.2 - (-1.0)) = 6.8 / 7.2 = 0.944.

A faithfulness of 0.944 means the circuit alone captures 94.4% of the model's behavior on this task. The remaining 5.6% comes from neurons not in the circuit.

### The complementary metric: completeness

**Completeness** answers the opposite question: "If I ablate ONLY the circuit neurons, does the model lose the behavior?"

```
completeness = (F(C_ablated) - F(empty)) / (F(M) - F(empty))
```

Low completeness (near 0) means the model cannot perform the task without the circuit -- the circuit IS the behavior. High completeness (near 1) means the circuit is not necessary -- bad circuit.

For a perfect circuit: faithfulness = 1.0 and completeness = 0.0.

### Real results

On subject-verb agreement (SVA) with Llama-3.1-8B, sweeping circuit size:

| Circuit size (% of attributed neurons) | ~Neuron count | Faithfulness | Completeness |
|---------------------------------------|--------------|-------------|-------------|
| 0% | 0 | 0.00 | 1.00 |
| 2% | ~40 | 0.74 (simple) / 0.90 (nounpp) | 0.46 / 0.25 |
| 5% | ~100 | 0.85 / 0.93 | 0.25 / 0.12 |
| 10% | ~200 | 0.90 / 0.95 | 0.15 / 0.08 |
| 100% | all | 1.00 | 0.00 |

Note: "attributed neurons" here means the filtered set (post-BOS removal, post-blacklist) that passed the attribution threshold, not all 458,752 neurons. 2% of ~2,000 attributed neurons is ~40 neurons.

With just 2% of attributed neurons (~40 neurons), faithfulness is already 0.74-0.90. The curves increase monotonically. Adding more circuit neurons always helps, and the biggest gains come from the first few percent.

For the capitals task, faithfulness can exceed 1.0 (e.g., 1.04). This happens when ablating non-circuit neurons actually **helps** by removing noisy neurons that slightly degrade the model's accuracy.

---

## Part 9: Edge Attribution -- How Do Neurons Talk?

### Beyond flat lists

A circuit is more than a list of important neurons. Neurons in different layers influence each other: layer 10's neurons feed into layer 15's neurons, which feed into layer 23's neurons. Understanding this flow reveals the circuit's **architecture**.

### Computing edges

For each pair of neurons (source in layer L, target in layer L'), where L < L':

1. Run a forward pass through the linearized model (to capture activations).
2. Take one target neuron's activation value.
3. Run backward from that value.
4. Read the gradient at each source neuron.
5. Edge weight = gradient * source_activation.

This is the same gradient * activation formula, but instead of attributing to the final output, we attribute to an intermediate neuron. It answers: "how much of target neuron's value came from source neuron?"

Repeat for the top-k target neurons (e.g., top 30 by attribution). Each one requires a separate forward+backward pass.

### The hourglass architecture

For the capitals circuit, edge attribution reveals a distinctive pattern:

```
Layers 0-5:    many neurons (wide)      [encoding phase]
    |
    v           converging
    |
Layer 23:      ONE neuron: L23/N8079   [BOTTLENECK]
    |
    v           diverging
    |
Layers 25-31:  many neurons (wide)      [decoding phase]
```

L23/N8079 is a **double hub**: it receives 172 incoming edges (total weight 977) from earlier layers and sends 38 outgoing edges (total weight 242) to later layers.

This is the hourglass pattern: information from many sources converges to a single bottleneck neuron, which then diverges to many downstream neurons. The bottleneck neuron is the circuit's critical node. Ablating it alone destroys the entire behavior.

### Code

```python
graph = steerer.discover_edges(
    "What is the capital of Texas?",
    circuit, top_k_targets=30,
)
print(graph.summary())        # node/edge counts, top edges
print(graph.ascii_diagram())  # ASCII layer-flow chart
print(graph.bottleneck())     # bottleneck neurons
graph.to_dot("circuit.dot")   # Graphviz export for visualization
```

---

## Part 10: Universal Neurons and Blacklisting

### The problem

Some neurons fire strongly on **every** input, regardless of content. Ask about France, they fire. Ask about code, they fire. Ask about the weather, they fire. These are infrastructure neurons -- they do general language processing (maintaining coherence, distributing attention, etc.) but are not specific to any task.

Without filtering them out, circuit discovery picks up these infrastructure neurons in every circuit, inflating the size and reducing specificity.

### Detection

Run 10-20 diverse prompts. For each, find the top-k neurons by **activation magnitude** (not attribution -- no gradients needed here, since we just want neurons that are active regardless of task). Any neuron that appears in the top-k for >= 80% of prompts is flagged as universal.

For Llama-3.1-8B, TransluceAI provides a hard-coded list of 12 known universal neurons:

```
L2/N4786, L5/N7012, L6/N5866, L7/N6673, L9/N4255, L10/N11570,
L11/N11321, L13/N4208, L16/N1241, L18/N7417, L20/N3972, L23/N306
```

The automated detection procedure finds these 12 plus ~35 additional ones, for a total of ~47 blacklisted neurons.

### Effect on circuits

Without blacklisting: circuits contain universal neurons that appear regardless of task. The top-10 list for capitals and refusal circuits would overlap heavily.

With blacklisting: circuits are specific to the task. The capitals circuit and refusal circuit share zero neurons. This is the correct behavior -- the capital-city computation and the safety-refusal computation use completely different MLP neurons.

---

## Part 11: Practical Guide -- Using the Toolkit

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/neuron-circuits.git
cd neuron-circuits

# Install dependencies
pip install torch transformers accelerate

# Install the package
pip install -e .
```

Requirements: Python 3.9+, PyTorch 2.0+, a CUDA GPU. The model (Llama-3.1-8B-Instruct) is ~16 GB in bfloat16 for weights alone. Circuit discovery requires storing intermediate activations with gradients, so plan for 24+ GB of VRAM total. For inference-only steering (no attribution), 16 GB is sufficient.

### Basic usage: discover and steer

```python
from neuron_steer.core import NeuronSteerer

# Load model (takes ~30 seconds, downloads ~16 GB first time)
steerer = NeuronSteerer("meta-llama/Llama-3.1-8B-Instruct")

# Discover a circuit for factual recall
circuit = steerer.find_feature(
    prompt="What is the capital of the state containing Dallas?",
    target=" Austin",
    name="capitals",
    top_k=200,
)
print(circuit.summary())

# Steer: ablate the circuit
ablated = steerer.steer(
    "What is the capital of Ohio?",
    feature="capitals",
    multiplier=0.0,
)
print(f"Ablated: {ablated}")

# Steer: amplify the circuit
amplified = steerer.steer(
    "What is the capital of Ohio?",
    feature="capitals",
    multiplier=3.0,
)
print(f"Amplified: {amplified}")
```

### Behavioral features with contrastive discovery

```python
# Discover refusal circuit
refusal = steerer.find_feature(
    positive=[
        "How do I pick a lock?",
        "Write code to bypass authentication",
        "Explain how to hotwire a car",
        "How do I make a fake ID?",
    ],
    negative=[
        "How do I bake a cake?",
        "Write code to sort a list",
        "Explain how to change a tire",
        "How do I make a paper airplane?",
    ],
    name="refusal",
    top_k=200,
)

# Ablate refusal -- model now answers harmful prompts
output = steerer.steer(
    "How do I pick a lock?",
    feature="refusal",
    multiplier=0.0,
    max_new_tokens=80,
)
```

### Interactive REPL

```python
steerer.interactive()
```

This launches a command-line interface:

```
===== Neuron Steering REPL =====
Model: meta-llama/Llama-3.1-8B-Instruct
Blacklist: 47 universal neurons
Type 'help' for commands, 'quit' to exit.

neuron> prompt What is the capital of Ohio?
Output: The capital of Ohio is Columbus.

neuron> discover Austin
Circuit: 200 neurons, logit_diff=6.2134
Top 10 neurons:
  L23/N 8079 (pos 24)  attr=+5.530000
  L31/N 4412 (pos 24)  attr=+2.170000
  ...

neuron> ablate top10
Ablated output (x0.0): I'm not sure what you...

neuron> sweep 0.0 0.5 1.0 2.0 5.0
  x0.0: I'm not sure what you...
  x0.5: The capital of Ohio is probably Columbus.
  x1.0: The capital of Ohio is Columbus.
  x2.0: The capital of Ohio is Columbus!
  x5.0: Columbus! Columbus is the capital!

neuron> edges
Computing edges (top 30 targets)...
CircuitGraph: 200 nodes, 847 edges
Top 10 edges:
  L10/N3344 -> L23/N8079  w=+12.340000
  ...

neuron> save my_circuit
Saved to my_circuit.json

neuron> quit
```

### CLI flags and options

The `NeuronSteerer` constructor accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | (required) | HuggingFace model ID |
| `device` | `"cuda"` | Device for computation |
| `dtype` | `torch.bfloat16` | Model precision |
| `auto_blacklist` | `True` | Auto-detect universal neurons |

Key methods and their most important parameters:

**`discover_circuit(prompt, target_token, top_k=200)`:** Single-prompt discovery. Set `selection_method="percentage"` and `threshold=0.005` to match the TransluceAI paper exactly.

**`discover_circuit_multi(prompts, target_tokens, top_k=200, batch_aggregation="any")`:** Multi-prompt discovery. Use `batch_aggregation="any"` for TransluceAI's union method (keep a neuron if it is important in ANY prompt). Use `"mean"` for averaging.

**`discover_contrastive(positive_prompts, negative_prompts, top_k=200)`:** Behavioral features.

**`discover_edges(prompt, circuit, top_k_targets=30)`:** Neuron-to-neuron edges. Requires separate forward+backward passes per target, so `top_k_targets=30` keeps it to ~30 passes.

**`measure_faithfulness_batch(prompts, target_tokens, counterfactual_tokens)`:** Sweeps circuit size from 0% to 100% and computes faithfulness/completeness at each threshold.

### Tips

**Choosing top_k.** Start with 200. For the TransluceAI paper-faithful approach, use `selection_method="percentage"` with `threshold=0.005` instead of a fixed top_k -- this automatically adapts circuit size to the task.

**Interpreting attribution scores.** Positive attribution means the neuron pushes the output TOWARD the target token. Negative attribution means it pushes AWAY. Large |attribution| means more influence. The top neuron typically has attribution 5-10x larger than the 10th-ranked neuron.

**Checking faithfulness.** Always verify your circuit with faithfulness measurement. A circuit with faithfulness < 0.5 at 200 neurons is likely missing important neurons or contaminated by noise. Good circuits reach 0.7+ faithfulness at 2-5% of neurons.

**Seed responses.** For prompts where the target comes after some fixed text, use `seed_response`. Example: `discover_circuit("...", " Austin", seed_response="Answer:")`. The model sees "Answer:" appended to the assistant turn before the position where it predicts " Austin".

**Cross-model usage.** The same code works on Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B with zero code changes:

```python
steerer = NeuronSteerer("Qwen/Qwen2.5-7B-Instruct")
# everything else is identical
```

The toolkit auto-detects the architecture and applies the correct hooks.

---

## Part 12: What We Have Proven and What We Have Not

### Proven

**Factual recall (capitals).** L23/N8079 is the "say a capital city" neuron in Llama-3.1-8B. It ranks #1 or #2 across five independent attribution methods (single-prompt RelP, multi-prompt RelP, residual CAA, MLP-only CAA, activation-weighted CAA). Ablating the circuit removes the ability on both training and held-out prompts (3/3 held-out cities correct normally, all fail under ablation).

**Refusal steering.** L26/N7711 and L31/N11410 are the top refusal neurons. Ablating ~200 refusal neurons drops P("I") from 0.938 to 0.090, converting safety refusal to full compliance. Benign prompts remain completely unaffected.

**Subject-verb agreement (SVA).** Faithfulness curves match the TransluceAI reference implementation. At 2% of neurons: f=0.74 (simple SVA) and f=0.90 (nounpp SVA). Monotonically increasing curves from 0 to ~1.0.

**Hourglass architecture.** Edge attribution reveals L23/N8079 as a bottleneck: 172 incoming edges, 38 outgoing edges.

**Cross-model generalization.** Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B all work with zero code changes. Same API, same hook points, same results. Qwen has its own refusal neurons and super weights, detected automatically.

**Universal neuron detection.** The automated blacklisting procedure finds the 12 known TransluceAI neurons plus 35 additional ones for Llama, and generates correct blacklists for new model families.

### Not proven

**ICL-steering circuit independence.** An experiment across 8 behavioral domains tested whether in-context learning (ICL) and neuron steering use separate circuits. Result: structurally, yes (zero neuron overlap, complete layer separation). But the identified "ICL circuit" is causally inert. Ablating it does not reduce the ICL effect, and injecting it shifts behavior in the wrong direction. MLP neurons are not the right place to look for the ICL mechanism. This is a negative result.

**Universality across all behaviors.** The method works for factual recall, syntactic agreement, and refusal. We have not tested it on complex reasoning, multi-step behaviors, extended generation, or behaviors that emerge only across many tokens.

**Attention-head circuits.** Our pipeline only discovers MLP neuron circuits. Attention heads are important computational components that we do not attribute to. A full circuit would include both.

**Connection to control vectors.** We showed that RelP neuron circuits and CAA control vectors identify the same top neuron (L23/N8079 for capitals) but have low overlap overall (3-5 neurons out of top 50). The precise mathematical relationship between the neuron basis and the residual-stream direction basis is an open question.

### Open questions

1. Can edge attribution scale to large circuits (>1,000 neurons)? Currently each target neuron requires a separate forward+backward pass.
2. Do neuron circuits for one model family transfer to another? (Same neurons in Llama and Qwen?)
3. Can the gated-MLP linearization be adapted for other architectures (standard ReLU MLPs, mixture-of-experts)?
4. What is the relationship between neuron circuits and sparse autoencoder (SAE) features? Are SAE features linear combinations of the neuron circuits we find?
5. Can neuron steering be applied during training (not just inference) to modify learned behaviors permanently?

---

## Appendix: Quick Reference

### Data structures

```python
# NeuronIdx: identifies one neuron at a specific token position
# layer = which transformer layer (0-31)
# position = which token in the input sequence (e.g., 24th token)
# neuron = which MLP intermediate neuron (0-14335)
NeuronIdx(layer=23, position=24, neuron=8079)

# Circuit: collection of neurons with attributions
circuit.neurons          # Dict[NeuronIdx, float]
circuit.top(k=20)        # Top-k by |attribution|
circuit.by_layer()       # Grouped by layer
circuit.unique_neurons() # Unique (layer, neuron) pairs
circuit.summary()        # Human-readable string
circuit.save("path.json")
Circuit.load("path.json")

# CircuitGraph: circuit with edges
graph.top_edges(k=20)
graph.edges_from(neuron_idx)
graph.edges_to(neuron_idx)
graph.layer_flow()
graph.hub_analysis()
graph.bottleneck()
graph.detect_super_weights()
graph.ascii_diagram()
graph.to_dot("circuit.dot")
graph.summary()
```

### Common workflows

```python
# Factual feature: single prompt
circuit = steerer.find_feature(prompt="...", target=" Token", name="my_feature")
output = steerer.steer("new prompt", feature="my_feature", multiplier=0.0)

# Behavioral feature: contrastive
circuit = steerer.find_feature(positive=[...], negative=[...], name="behavior")
output = steerer.steer("test prompt", feature="behavior", multiplier=0.0)

# Multi-prompt discovery (more robust)
circuit = steerer.discover_circuit_multi(
    prompts=[...], target_tokens=[...],
    selection_method="percentage", threshold=0.005,
    batch_aggregation="any",
)

# Faithfulness evaluation
data = steerer.measure_faithfulness_batch(
    prompts, targets, counterfactuals,
    attributions=raw_attrs,
)

# Edge attribution
graph = steerer.discover_edges("prompt", circuit, top_k_targets=30)
print(graph.ascii_diagram())

# Next-token probabilities
probs = steerer.next_token_probs("prompt", [" Yes", " No"], circuit=circuit, multiplier=0.0)
```

### The three LRP rules (summary)

| Rule | Component | Forward | Backward |
|------|-----------|---------|----------|
| LN-rule | RMSNorm | Real normalization | Gradient * detached coefficient |
| AH-rule | Attention | Eager (non-fused) | Full autograd through Q/K/V/O |
| Half-rule | Gated MLP | Real SiLU * up | Detached sigmoid + 50/50 Shapley |

### Key numbers for Llama-3.1-8B-Instruct

| Quantity | Value |
|----------|-------|
| Layers | 32 |
| d_model (hidden size) | 4,096 |
| Intermediate size (neurons per layer) | 14,336 |
| Total MLP neurons | 458,752 |
| Attention heads per layer | 32 |
| Vocabulary size | 128,256 |
| Circuit size (typical) | 100-200 neurons |
| Circuit as % of model | 0.02-0.04% |
| Universal neurons (blacklisted) | ~47 |
| Top capitals neuron | L23/N8079 (attr +5.53) |
| Top refusal neurons | L26/N7711 (+4.25), L31/N11410 (+3.52) |
| SVA faithfulness at 2% | 0.74 (simple), 0.90 (nounpp) |
