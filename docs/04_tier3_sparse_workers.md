# TBI Whitepaper Appendix D: Conditional Depth, Specialization, and Memory Hierarchy

> **Document:** `docs/04_tier3_sparse_workers.md`  
> **Depends on:** [docs/03_tier2_routing.md](03_tier2_routing.md)  
> **Related:** [docs/05_background_daemons.md](05_background_daemons.md)

---

## Overview

This appendix addresses the heavy path of TBI: what happens after the routing layer decides a request requires more computation. The central mechanisms are **conditional depth**, **expert specialization**, and, in the longer term, **finer-grained memory hierarchy control**.

The most important scientific revision in this whitepaper is the separation between:

- mechanisms that are feasible today on commodity accelerators
- mechanisms that are better treated as future hardware research

This distinction is essential for credibility.

---

## 1. Feasible Today on Commodity Accelerators

Three families of techniques are practical today.

### 1.1 Early Exit at Fixed Checkpoints

A model may expose one or more exit checkpoints. At checkpoint $n$, a lightweight exit head evaluates whether continued computation is likely to change the answer enough to justify additional cost.

Let $h_n$ be the hidden state at checkpoint $n$ and let $q_n(h_n)$ be an exit head. A generic early-exit rule is:

$$
n^{\ast}(x) = \min\left\{\, n \in \mathcal{G} \;\middle|\; \kappa_n(x) \geq \tau_n \,\right\}
$$

where:

- $\mathcal{G}$ is the set of checkpoint layers
- $\kappa_n$ is a confidence score
- $\tau_n$ is a calibrated threshold

If no checkpoint satisfies the rule, the model continues to the final layer.

### 1.2 Expert or Domain Specialization

A serving system may keep several resident experts or models specialized for domains such as code, mathematics, or customer support. Routing can choose among them based on input features. This is realistic today and already supported by several serving patterns.

### 1.3 Multi-Model Cascades

Instead of trying to swap parameter blocks in and out at very fine granularity, current systems often obtain most of the benefit by hosting multiple preloaded models:

- a small model
- a medium model or shallow path
- a full model

This is a strong near-term baseline and should be considered the default TBI path on commodity hardware.

---

## 2. Early Exit as a Cost-Quality Trade-off

### 2.1 Exit-Head Overhead

Early exit is attractive only when the checkpointing overhead is smaller than the saved computation. If the exit head is too large or too frequent, it can erase the benefit.

Therefore:

- checkpoints should be sparse
- exit heads should be low-cost
- thresholds should be calibrated on real workloads

### 2.2 Batching Interaction

In batched serving, different sequences may satisfy exit criteria at different depths. This complicates scheduling and can reduce effective savings: in continuous batching, sequences that exit early create fragmented KV cache occupancy, and remaining sequences compete for the same memory with partially vacated but not yet deallocated slots, reducing effective GPU utilization across the batch.

A strong TBI implementation must therefore report a **batch efficiency correction factor** $\rho_{\text{batch}} \in (0, 1]$, defined as the ratio of realized batch-level energy savings to the theoretical per-request savings:

$$\Delta J_{\text{realized}} = \rho_{\text{batch}} \cdot \Delta J_{\text{theoretical}}$$

Evaluation tables must report $\rho_{\text{batch}}$ at the representative batch sizes used in production. Single-request theoretical savings without this correction factor are insufficient for publication-grade claims.

### 2.3 Calibration Requirement

Early-exit confidence is not the same as correctness. The scientific failure mode is overconfident early answers on hard examples. For this reason, early-exit systems should be evaluated with:

- quality loss relative to full-depth decoding
- calibration error
- realized savings under batch load
- failure concentration on hard subsets

---

## 3. Specialization

### 3.1 Why Specialization Matters

If traffic has strong domain structure, specialized experts can improve both quality and efficiency:

- better quality per parameter in the target domain
- higher chance of confident early termination
- cheaper serving than forcing every request through a universal path

### 3.2 Serving Interpretation

The key practical point is that specialization need not imply exotic hardware. It may simply mean:

- multiple resident models
- MoE-style parameter subsets
- domain-conditioned adapters or LoRA stacks
- expert-specific shallow exits

TBI should frame these as existing engineering options rather than entirely novel mechanisms.

---

## 4. Future Hardware Research Direction

The original TBI concept includes a more aggressive idea: parameter blocks with explicit hot, warm, and cold states across SRAM, HBM, and non-volatile storage.

This is scientifically interesting, but it should be framed as a **future accelerator research direction**, not as a default assumption for present-day GPUs.

### 4.1 Why the Idea Is Interesting

Fine-grained residency control could, in principle:

- lower idle memory cost
- reduce thermal floor
- allow more dynamic capacity sharing between active weights and context state
- expand the set of conditional-compute strategies

### 4.2 Why It Is Not a Commodity-Hardware Claim

Current accelerators generally do not expose:

- application-level control of on-chip SRAM residency for large parameter blocks
- per-request movement of multi-gigabyte blocks into SRAM
- predictable latency for non-volatile parameter paging at interactive serving scale

As a result, any claim about multi-gigabyte hot, warm, and cold parameter orchestration should be labeled speculative unless backed by hardware-specific evidence.

### 4.3 Recommended Wording

A publication-grade TBI paper should say:

> Fine-grained parameter residency control is a promising co-design direction for future accelerators. This whitepaper does not assume that present-day commodity accelerators expose that control at the granularity required by the most aggressive TBI memory-hierarchy proposals.

That wording is much stronger than pretending the capability already exists.

---

## 5. Practical TBI Memory Hierarchy Today

On current systems, the realistic memory story is:

- active models or experts are typically resident in device memory during service windows
- host-to-device and storage-to-device movement occurs at coarse granularity
- retrieval state, KV cache, and batch scheduling dominate many latency trade-offs
- compression and specialization are often more practical than aggressive parameter paging

This means the near-term TBI program should prioritize:

- resident model cascades
- expert or adapter selection
- early exit
- KV and retrieval efficiency
- quantization and distillation

---

## 6. Research Questions for Custom Hardware

The future hardware branch of TBI remains valuable if it is posed as a research agenda.

Key questions include:

1. Can parameter blocks be placed into multiple residency states with predictable latency and energy behavior?
2. Can route-aware schedulers exploit such states without destroying interactivity?
3. Does fine-grained residency control improve quality-per-joule beyond what cascades and early exit already achieve?
4. What memory-controller and compiler support would be required?

These are legitimate hardware-systems questions and could support an accelerator paper on their own.

---

## 7. Evaluation Criteria

A strong TBI heavy-path evaluation should report:

- quality relative to a full-depth baseline
- energy and latency distributions
- realized savings under batch load
- scheduling overhead
- gains from specialization
- the gap between commodity-hardware implementations and hardware-co-design simulations, if both are studied

This makes it possible to separate what TBI proves today from what it proposes for the future.

---

## 8. Summary

The heavy-compute layer of TBI is most credible when described in two tiers. The first tier is practical today: early exit, specialization, resident multi-model cascades, and compression. The second tier is future-facing: fine-grained parameter residency and memory-state control on custom accelerators. Keeping those claims separate strengthens the whole project.
