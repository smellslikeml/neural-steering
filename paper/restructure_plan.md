# Paper Restructure Plan: From Reproduction Study to Original Research

**Authors**: Paleas, Karan Malhotra (@karan4d), Sam Herring (@yaboilyrical) — Nous Research
**Date**: 2026-02-17
**Current title**: "Reproducing and Extending Neuron-Basis Circuit Discovery: A Practical Guide"
**Problem**: The current paper frames itself as a reproduction study with extensions. The actual contributions are significantly more original than the framing suggests. We need to reposition as original research that builds on Arora et al.'s foundation.

---

## 1. Title Options

### Option A (Recommended): "Contrastive Neuron Circuits: Behavioral Steering and Cross-Scale Analysis in the Neuron Basis"
- Leads with the most novel contribution (contrastive discovery)
- Signals the breadth of analysis (cross-scale)
- "Neuron basis" connects to Arora et al. without being subordinate to it

### Option B: "From Sparse Circuits to Behavioral Steering: Neuron-Level Intervention Across Model Scales"
- Narrative arc: discovery → intervention → scaling
- Positions the work as a progression from foundational finding to practical application
- "Behavioral steering" signals the applied value

### Option C: "Neuron-Level Behavioral Circuits: Contrastive Discovery, Layer Localization, and Scaling Laws"
- Most descriptive; lists the three pillars of new experiments
- "Scaling laws" is attention-grabbing if the 1B vs 8B results show clear patterns
- Risk: might oversell if scaling results are preliminary

---

## 2. New Abstract

> Arora et al. (2026) demonstrated that language model circuits are remarkably sparse in the neuron basis: roughly 100–200 MLP neurons, identified via Relevance Propagation (RelP), form faithful circuits for factual recall and syntactic agreement tasks. We build on this foundation with three lines of original work. First, we introduce **contrastive neuron discovery**, which applies Contrastive Activation Addition at the MLP neuron level, enabling circuit discovery and targeted behavioral steering for arbitrary behaviors — including refusal, belief, and sentiment — where single-token attribution targets are unavailable. This bridges two previously separate paradigms: residual-stream steering and neuron-level circuit analysis. Second, we conduct a **cross-scale circuit analysis** comparing circuit structure between 1B and 8B parameter models, examining whether circuits compress, expand, or reorganize across scales — the first such comparison using RelP-based attribution. Third, we perform **layer localization and circuit overlap studies**, mapping where behavioral circuits concentrate across the network depth and measuring whether circuits for different behaviors share neuronal infrastructure or remain independent. We also contribute an automated universal neuron detection procedure that replaces hand-curated blacklists, making the pipeline immediately portable to new models. Our analysis reveals [hourglass topology / layer concentration patterns / overlap statistics — to be filled after experiments]. All experiments use a standalone toolkit requiring only PyTorch and Hugging Face Transformers, with zero code changes across four model families spanning 1B to 8B parameters.

*(Bracketed text to be updated with actual experimental findings.)*

---

## 3. Section-by-Section Outline

### Section 1: Introduction

**Current problem**: Opens with general mech interp, then positions as reproduction.
**New framing**: Open with the *gap* in the field, not with the method we're reproducing.

**Outline**:

1. **The circuit discovery landscape** (2 paragraphs)
   - Mech interp seeks circuits. Three representational bases exist: residual stream, attention heads, SAE features. Each has trade-offs.
   - Arora et al. (2026) demonstrated a compelling fourth option: the neuron basis. ~100-200 MLP neurons form faithful circuits. This is foundational — but their work focused on single-token attribution tasks (factual recall, SVA). Several questions remain open.

2. **The open questions** (1 paragraph, bulleted)
   - How do we discover circuits for behaviors that don't have clean single-token targets (refusal, belief, sentiment)?
   - Do circuits for different behaviors share neurons, or are they independent?
   - Where in the network do behavioral circuits concentrate? Is belief localized?
   - How does circuit structure change across model scales?
   - Can neuron-level circuits provide a more natural decomposition of steering vectors than SAEs?

   *Note*: Frame 2-3 of these as directly connecting to questions raised by Goodfire (Chalnick et al., 2025) — they asked "could we causally intervene on specific neurons?" and "does this suggest that belief is localized to these layers?" We answer both.

3. **This paper** (1 paragraph + enumerated contributions)
   - We address these questions through five contributions:
     1. **Contrastive neuron discovery** — CAA at the neuron level, bridging steering and circuits
     2. **Layer localization analysis** — mapping behavioral circuits across network depth
     3. **Circuit overlap/independence** — measuring shared vs. task-specific neural infrastructure
     4. **Cross-scale circuit analysis** — comparing 1B and 8B parameter models
     5. **Automated universal neuron detection** — replacing hand-curated blacklists for portability
   - We also introduce a new behavioral task (belief/sentiment steering) and report observations on hourglass topology and super weight connections.

**Framing language**: "Building on the foundational discovery of Arora et al. (2026) that circuits are sparse in the neuron basis, we..." — NOT "We reproduce and extend..."

### Section 2: Background and Related Work (merged, shortened)

**Rationale**: The current paper has separate Background and Related Work sections that are both long. Merge them. The background should be just enough to make the paper self-contained.

**Subsections**:
- 2.1 **Neuron-Basis Circuit Discovery** — Brief summary of Arora et al.'s method (RelP + 3 rules). 1 paragraph + equations. Cite Jafari et al. for validation.
- 2.2 **Contrastive Activation Addition** — CAA in the residual stream. The gap: CAA operates on the full residual stream, not individual neurons.
- 2.3 **Sparse Autoencoders and Their Limitations** — SAEs as an alternative representational basis. Cite Mayne et al. (2024): SAEs can't decompose steering vectors. This motivates neuron-basis analysis.
- 2.4 **Open Questions from Neuron-Level Interpretability** — Cite Goodfire's questions about causal intervention and belief localization. Frame our work as answering these.

**What to cut**: Super weights subsection moves to discussion (it's an observation, not background). Detailed LRP history moves to a single citation.

### Section 3: Method

**Current problem**: Every subsection says "from Arora et al." This frames us as implementers, not researchers.

**New structure**:

- 3.1 **Foundation: RelP Attribution in the Neuron Basis**
  - 1 paragraph: "We build on the RelP attribution pipeline of Arora et al. (2026), which we briefly summarize." Then the 3 rules in compressed form (keep equations, drop verbose explanations). Algorithm box stays but caption changes to just "Neuron Circuit Discovery via RelP (Arora et al., 2026)" without "from" emphasis.
  - Key: This is *background context*, not a contribution. Keep it short and move on.

- 3.2 **Contrastive Neuron Discovery** ← STAR CONTRIBUTION, gets full treatment
  - Motivation: Many interesting behaviors (refusal, belief, sentiment, sycophancy) cannot be expressed as single-token attribution targets. RelP requires backward from a specific logit. CAA doesn't require a target token but operates in the full residual stream.
  - Method: We bridge these by applying contrastive activation analysis at the MLP neuron level.
  - Algorithm box: Formal description of contrastive discovery
  - Key insight: "This is essentially CAA applied at the MLP neuron level rather than the residual stream. The resulting circuits inherit the sparsity of neuron-basis circuits (~200 neurons, 0.04% of MLP) while capturing behaviors that RelP alone cannot target."
  - **Connection to the field**: "Mayne et al. (2024) showed that SAEs cannot faithfully decompose residual-stream steering vectors. Our contrastive neuron circuits provide an alternative decomposition that operates in the natural neuron basis rather than a learned feature dictionary."

- 3.3 **Automated Universal Neuron Detection**
  - Current Section 3.3 content, but reframed: not "engineering convenience" but "enabling portability" — this is what makes the pipeline work on new models without manual intervention.
  - Add: comparison table showing detected vs. hand-curated neurons for Llama-3.1-8B to validate the procedure.

- 3.4 **Steering via Targeted Activation Modification**
  - Current Section 3.6, shortened. The key point: we modify 0.04% of neurons, not the entire residual stream.

- 3.5 **Edge Attribution and Circuit Topology Analysis**
  - Current Sections 3.5 and 3.6, merged and shortened. Frame as an analytical tool we use to understand circuit structure.

### Section 4: Experimental Setup

Short section. Hardware, models, hyperparameters, prompt datasets.

**Models**: Llama-3.1-8B-Instruct (primary), Llama-3.2-1B-Instruct (scaling), Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (portability).

**Behavioral tasks**:
- Factual recall (capitals) — from Arora et al., used for validation
- Subject-verb agreement (SVA) — from Arora et al., used for faithfulness
- Refusal — contrastive discovery (our extension)
- Belief/sentiment — contrastive discovery (new task, our contribution)
- [Additional tasks from task #5 design]

### Section 5: Results — Validation of Foundation

**Purpose**: Establish that our implementation is correct, that the foundational claims hold, and THEN build on them. This section is short — it's not the point of the paper.

- 5.1 **Faithfulness Reproduction** — Current Section 4.3, compressed. Table + figure stay. 1 paragraph of text. "We confirm the core finding of Arora et al.: faithfulness reaches 0.74–0.90 at 2% circuit size."
- 5.2 **Cross-Model Portability** — Current Section 4.5, shortened. The key result: zero code changes across 4 model families.

### Section 6: Results — Contrastive Behavioral Circuits (NEW — OUR CORE CONTRIBUTION)

- 6.1 **Refusal Circuit Discovery and Steering**
  - Current Section 4.2, but expanded and reframed.
  - Not just "we ablated refusal" — instead: "Contrastive discovery identifies a sparse refusal circuit. Ablation converts refusal to compliance while preserving unrelated capabilities. This demonstrates that behaviors without single-token targets can be isolated at the neuron level."
  - Add: multiplier sweep results (from task #4). Show the smooth/sharp behavioral transition curve.
  - Add: specificity analysis (benign prompts unaffected).

- 6.2 **Belief/Sentiment Steering** (NEW — from task #5)
  - Full presentation of the new behavioral task.
  - Circuit discovery, steering results, specificity tests.
  - Connection to Goodfire: "Can we causally intervene on specific neurons to have predictable impacts on model behavior? Yes."

- 6.3 **Multiplier Sweep: Behavioral Phase Transitions**
  - From task #4 results.
  - Is the transition smooth or sharp? What does this tell us about how behaviors are encoded?
  - Figure: behavioral metric vs. multiplier alpha for refusal and belief tasks.

### Section 7: Results — Circuit Structure Analysis (NEW)

- 7.1 **Layer Localization: Where Do Behavioral Circuits Live?**
  - From task #1 results.
  - Layer distribution heatmaps for each behavior.
  - Do different behaviors concentrate in different layers? Is refusal in late layers? Is factual recall in middle layers?
  - Direct answer to Goodfire's question: "Does this suggest that belief is localized to these layers?"
  - Figure: layer distribution heatmap (behaviors × layers, color = neuron density)

- 7.2 **Circuit Overlap and Independence**
  - From task #2 results.
  - Jaccard similarity matrix between all behavioral circuits.
  - Identification of shared infrastructure neurons vs. task-specific neurons.
  - Key finding: Are circuits truly modular? Or is there a shared backbone?
  - Figure: overlap heatmap. Table: shared neurons identified.

- 7.3 **Cross-Scale Analysis: 1B vs. 8B**
  - From task #3 results.
  - Do circuits compress at smaller scale? Do they use different relative layers?
  - Does the hourglass pattern persist?
  - Figure: side-by-side layer distributions at 1B and 8B. Table: circuit size comparison.

- 7.4 **Hourglass Topology and Super Weight Connections**
  - Current Section 4.4, but now with multi-task data.
  - Does the hourglass pattern appear across different behaviors? Different models?
  - Super weight connection: are the same neurons always source hubs?

### Section 8: Discussion

- 8.1 **CAA–RelP Complementarity** — Current Section 5.1, expanded with new behavioral tasks. The intersection of CAA and RelP is the strongest signal.
- 8.2 **Implications for SAE-Based Interpretability** — If neuron-basis circuits capture behaviors that SAEs struggle with (Mayne et al.), when should we use neurons vs. SAE features?
- 8.3 **Answering Goodfire's Questions** — Explicit section connecting our findings to the questions raised by feature-based interpretability work.
- 8.4 **Steering Precision and Safety Implications** — Modifying 0.04% of neurons achieves behavioral control. What does this mean for alignment? For jailbreaking risk?
- 8.5 **Limitations** — Condensed from current Section 6. Add limitations of new experiments.

### Section 9: Conclusion

- Summary of contributions (not "reproduction" — "we addressed open questions")
- What this means for the field
- Future work: more behaviors, more scales, real-time circuit monitoring

### Appendix

- A: Detailed LRP Rule Descriptions (move from Section 3 — keeps the main paper focused)
- B: Toolkit API Reference (current Appendix A)
- C: Universal Neuron Blacklists (current Appendix B)
- D: Full Prompt Datasets (current Appendix C + new prompts)
- E: Extended Results Tables

---

## 4. New Figures and Tables to Add

### New Figures

| Figure | Description | Source |
|--------|-------------|--------|
| **Layer Heatmap** | Behaviors × Layers heatmap showing neuron density per layer for each behavioral circuit. Color intensity = fraction of circuit neurons at that layer. | Task #1 results |
| **Circuit Overlap Matrix** | Pairwise Jaccard similarity heatmap for all behavioral circuits. Off-diagonal values show shared structure. | Task #2 results |
| **Cross-Scale Comparison** | Side-by-side layer distribution plots for 1B and 8B models on the same tasks. Shows whether circuits shift, compress, or expand. | Task #3 results |
| **Multiplier Sweep Curves** | Behavioral metric (P("I") for refusal, correctness for capitals, belief score for belief task) vs. multiplier alpha from 0.0 to 3.0. Shows transition shape (smooth vs. sharp). | Task #4 results |
| **Belief Steering Demo** | Example generations showing belief/sentiment being steered via circuit ablation/amplification. Before/after comparison. | Task #5 results |
| **Venn Diagram of Paradigms** | Conceptual figure showing how contrastive neuron discovery bridges CAA (behavioral steering) and RelP (circuit attribution). Our method sits at the intersection. | Manual creation |

### New Tables

| Table | Description | Source |
|-------|-------------|--------|
| **Behavioral Circuit Summary** | For each behavior: circuit size, key neurons, layer concentration, steering effect. Master table of all circuits. | All experiments |
| **Overlap Statistics** | Pairwise Jaccard scores between circuits, plus list of shared infrastructure neurons. | Task #2 |
| **Scaling Comparison** | Circuit size, faithfulness, layer distribution, bottleneck presence for 1B vs 8B. | Task #3 |
| **Multiplier Sweep Data** | Full sweep results (alpha, behavioral metric, perplexity delta) for all tasks. | Task #4 |
| **Belief/Sentiment Results** | Steering results for the new behavioral task. | Task #5 |

### Figures to Keep (Modified)

- **Faithfulness curves** (Figure 1 currently) — Keep but make smaller, move earlier as validation.
- **Hourglass topology** (Figure 2 currently) — Keep but expand to show multi-task comparison if available.
- **Hub analysis table** — Keep, expand to multiple tasks.

---

## 5. Framing the Relationship to Arora et al.

### Principle: Foundation, Not the Whole Story

Arora et al. showed that circuits are sparse in the neuron basis. This is a *foundational discovery* — analogous to showing that a particular representational basis has useful properties. Our work takes that foundation and asks: **what can we do with it?**

### Specific Framing Language

**Introduction** (establishing the relationship):
> "Arora et al. (2026) demonstrated the foundational result that language model circuits are remarkably sparse in the neuron basis. We build on this discovery to address several open questions: ..."

**Method** (the foundation section):
> "We adopt the RelP attribution pipeline of Arora et al. (2026) as our circuit discovery foundation. We briefly summarize their three linearization rules for completeness; the reader is referred to their paper for the full derivation."

**Results — Validation** (confirming the foundation):
> "Before presenting our novel analyses, we confirm that our implementation reproduces the core findings of Arora et al. (2026). This validation establishes the reliability of our experimental platform."

**Results — Original work** (building on it):
> "With the foundation validated, we now turn to the questions that the original work leaves open: ..."

**Discussion** (the broader picture):
> "Where Arora et al. (2026) established *that* circuits are sparse in the neuron basis, our work begins to characterize *how* these circuits are organized: their layer distribution, inter-circuit relationships, and scaling properties."

### What NOT to say

- ~~"This paper is a reproduction study"~~
- ~~"The core method is entirely theirs"~~
- ~~"Our aim is to make this method more accessible"~~
- ~~"as an engineering convenience"~~
- ~~"We report this as an empirical observation from running Arora et al.'s edge attribution"~~

### What TO say instead

- "Building on Arora et al.'s foundational result, we..."
- "We adopt their RelP attribution as our circuit discovery platform and extend it with..."
- "Our contrastive discovery method bridges two previously separate paradigms..."
- "This analysis reveals a consistent pattern across tasks and models..."

---

## 6. Connecting to the Broader Mech Interp Conversation

### 6.1 The SAE Debate

**Setup**: SAEs are the dominant paradigm for interpretable features. But they have known limitations:
- Expensive training
- Dictionary-granularity tradeoff
- Cannot faithfully decompose steering vectors (Mayne et al., 2024)
- Features are learned, not intrinsic — different SAE training runs yield different features

**Our position**: The neuron basis is *intrinsic* to the model. Arora et al. showed it's sparse enough to be useful. We show it's versatile enough to capture diverse behaviors. This isn't "neurons vs. SAEs" — it's "neurons provide a complementary basis that avoids SAE limitations for certain analyses."

**Framing language**: "While sparse autoencoders learn interpretable features via dictionary learning, the neuron basis offers an intrinsic alternative that requires no training and provides a canonical decomposition. Our results suggest that for behavioral circuit analysis — particularly for steering — the neuron basis may be more natural than learned feature dictionaries."

### 6.2 The Goodfire Connection

Goodfire (Chalnick et al., 2025) used SAE features to study concepts like "belief" and found they concentrate in specific layers. They asked:
- "Could we causally intervene on specific neurons in these layers to have predictable impacts on model behavior?" → **Our contrastive steering directly answers this: YES.**
- "Does this suggest that belief is localized to these layers?" → **Our layer localization study provides quantitative evidence.**
- "Are concept likelihood functions implemented in a non-linear way, and are they implemented in earlier or later layers?" → **Our edge attribution and layer analysis address this.**

**Framing language**: "Our layer localization analysis addresses questions raised by feature-based interpretability research (Chalnick et al., 2025): we provide direct evidence for [concentration/distribution] of behavioral circuits across network depth, and demonstrate that causal intervention on neuron-level circuits produces predictable behavioral shifts."

**Important caveat**: We need the actual Goodfire paper citation. If it's a blog post or technical report, cite it appropriately. Don't over-cite a blog post as if it's a peer-reviewed paper.

### 6.3 Circuit Discovery Method Landscape

Position our work in the broader landscape:

| Method | Basis | Cost | Training | Behaviors |
|--------|-------|------|----------|-----------|
| Activation Patching | Residual stream | O(n) forward passes | None | Any (but expensive) |
| ACDC | Attention + MLP | Many forward passes | None | Any (but expensive) |
| SAE Circuits | Learned features | 1 forward-backward | SAE training | Limited by dictionary |
| RelP (Arora) | Neurons | 1 forward-backward | None | Single-token targets |
| **Ours** | Neurons | 1 forward-backward | None | **Any (via contrastive)** |

This table makes our contribution clear: we extend the neuron-basis approach to arbitrary behaviors at the same computational cost.

### 6.4 Representation Engineering Bridge

Our contrastive neuron discovery is the natural decomposition of CAA steering vectors into the neuron basis. If you have a residual-stream steering vector, you can ask: which neurons implement it? Our method provides the answer. This bridges representation engineering (top-down, behavioral) with mechanistic interpretability (bottom-up, structural).

---

## 7. Section-by-Section Framing Language

### Introduction
- **Tone**: Confident but not overclaiming. We address open questions; we don't solve interpretability.
- **Key phrase**: "Building on the foundational discovery that circuits are sparse in the neuron basis (Arora et al., 2026), we present [number] lines of original analysis..."
- **Avoid**: "reproduction", "practical guide", "reimplementation", "for completeness"

### Background
- **Tone**: Efficient and positioned. Not a literature survey — just enough to make the paper self-contained.
- **Key phrase**: "The neuron basis offers a training-free alternative to SAE-based circuit discovery, but several questions about its scope and scaling properties remain unaddressed."

### Method — Foundation
- **Tone**: Brief and referential. "Arora et al. showed X; we use their method."
- **Key phrase**: "We adopt the RelP attribution pipeline of Arora et al. (2026) as our base. We briefly summarize for completeness and refer the reader to their work for full derivation."
- **Length**: 1 page max. Move detailed equations to appendix.

### Method — Contrastive Discovery
- **Tone**: This is where we claim novelty. Be specific about what's new.
- **Key phrase**: "We introduce contrastive neuron discovery, which applies the contrastive logic of CAA (Rimsky et al., 2024) at the MLP neuron level rather than the residual stream. This bridges behavioral steering and circuit-level analysis, enabling discovery for behaviors — refusal, belief, sentiment — where clean target tokens are unavailable."
- **Important caveat**: Be honest that this is conceptually simple (mean activation difference). The novelty is in the *bridging* of paradigms and the *experimental validation*, not algorithmic complexity.

### Results — Validation
- **Tone**: Quick and confident. "Our implementation reproduces the core findings."
- **Key phrase**: "Consistent with Arora et al. (2026), we observe faithfulness of 0.74–0.90 at 2% circuit size."
- **Length**: 1 page max.

### Results — Contrastive Circuits
- **Tone**: This is the showcase. Detailed, with multiple experiments.
- **Key phrase**: "Contrastive neuron discovery identifies sparse circuits for [refusal, belief, sentiment] that enable targeted behavioral steering while modifying only 0.04% of MLP neurons."

### Results — Structure Analysis
- **Tone**: Exploratory but data-driven. Present findings, note caveats.
- **Key phrase**: "Our layer localization analysis reveals that [finding]. Circuit overlap analysis shows [finding]. Cross-scale comparison demonstrates [finding]."
- **Important**: If results are weak or inconclusive on any axis, say so honestly. Don't stretch.

### Discussion
- **Tone**: Connecting findings to the broader field. What does this mean?
- **Key phrases**:
  - "These results suggest that the neuron basis provides a viable alternative to SAE features for behavioral circuit analysis..."
  - "Our layer localization findings directly address questions raised by [Goodfire]..."
  - "The low overlap between behavioral circuits suggests [modularity/independence]..."

### Limitations
- **Tone**: Honest and specific. Don't hide weaknesses.
- **Must include**: Contrastive circuits lack formal faithfulness evaluation. Cross-scale analysis is limited to two model sizes. Belief steering is a single task.

---

## 8. Critical Self-Assessment (per CLAUDE.md principles)

### Is this actually novel? What's the delta over prior work?

**Genuinely novel**:
- Contrastive neuron discovery (nobody has bridged CAA and neuron circuits)
- Cross-scale circuit comparison (nobody has compared RelP circuits across model sizes)
- Layer localization of behavioral circuits (new experimental contribution)
- Circuit overlap analysis (new experimental contribution)
- Automated universal neuron detection (practical, enables portability)

**Not novel, but valuable**:
- Cross-model runs (portability demonstration, not a research contribution)
- Hourglass observation (interesting but preliminary; needs more evidence to be a real claim)
- Super weight connection (observation, not a proven result)

**Honest assessment**: The contrastive discovery method is conceptually simple (mean activation difference). The novelty is in the *bridging* of two paradigms and the *experimental program* that follows. We should not oversell the method and instead let the experimental results speak.

### What claims need hedging?

- "Any behavior" — hedge to "a range of behaviors including X, Y, Z"
- Hourglass topology — keep as "preliminary observation" unless multi-task multi-model evidence is strong
- Super weight connection — "suggestive" not "demonstrated"
- Layer localization claims should be specific to our tested behaviors, not generalized

### Should this exist?

YES. The field needs:
1. Methods that work for behaviors beyond single-token targets (our contrastive approach)
2. Understanding of how circuits relate to each other (our overlap analysis)
3. Understanding of how circuits change across scale (our scaling analysis)
4. Practical tools that work on new models without retraining (our automated pipeline)

The paper should exist as original research if and only if the experimental results are strong. The method section alone (contrastive discovery + automation) is a minor contribution. The method + comprehensive experimental analysis = a real paper.

---

## 9. Action Items After Plan Approval

1. Wait for experimental results from tasks #1-5
2. Restructure the LaTeX following this plan
3. Write new abstract with actual numbers
4. Create new figures from experimental data
5. Add Goodfire citation (need to find the actual paper/report)
6. Have adversarial review before submission

---

## 10. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Experimental results are weak/inconclusive | Present honestly as exploratory; focus paper on contrastive discovery method |
| Layer localization shows no clear pattern | Report null result; discuss what this means (maybe behaviors are distributed, not localized) |
| Circuit overlap is near-zero everywhere | This is actually interesting — means circuits are truly modular. Frame positively. |
| Cross-scale results are noisy | Limit claims to "preliminary observations at two scales" |
| Reviewers say contrastive discovery is too simple | Acknowledge simplicity; argue that the contribution is the bridging + validation, not algorithmic novelty |
| Goodfire paper is not published/citable | Cite as blog post or remove connection; don't build the paper around it |
