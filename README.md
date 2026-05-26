# Thermodynamically Bounded Intelligence

_Open Technical Specification_

> **Status:** Research Draft  
> **Type:** Technical Proposal  
> **Domain:** AI Systems Architecture · Hardware-Aware ML · Efficient Inference

## Motivation

The mid-2020s mark a structural inflection point. The growth in AI compute demand is increasingly constrained by physical deployment limits: power availability, cooling capacity, memory bandwidth, latency targets, and thermal safety margins.

Thermodynamically Bounded Intelligence (TBI) frames this as a systems-design problem: build and update intelligence systems under explicit physical budgets rather than optimize task quality in isolation.

## Abstract

Large-scale machine learning systems are increasingly constrained not only by data and compute availability, but by deployment-time physical limits: energy draw, cooling capacity, memory bandwidth, latency targets, and thermal safety margins. Standard training objectives optimize task quality directly, while these deployment costs are usually handled later through engineering heuristics such as quantization, pruning, distillation, batching, and serving-time routing. This separation is practical, but incomplete: it leaves substantial efficiency gains unavailable to objectives that never observe deployment costs.

This whitepaper presents **Thermodynamically Bounded Intelligence (TBI)** as a research program for **energy-constrained adaptive inference**. TBI does not assume that silicon temperature or power draw are exactly differentiable with respect to model weights during ordinary training. Instead, it proposes a more defensible approach: deployment telemetry is used to build surrogate cost models and control policies that influence routing, conditional depth, expert selection, compression, and periodic retraining. In this formulation, hardware cost becomes a first-class systems objective without overclaiming exact end-to-end thermodynamic backpropagation.

The TBI proposal has two parts. The first is a **near-term software path** that is feasible on present-day accelerators: telemetry-informed routing, early exit, model cascades, expert specialization, distillation, quantization, and offline compression loops. The second is a **long-term hardware research path** involving finer-grained control of memory hierarchy and parameter residency across SRAM, HBM, and non-volatile storage. This whitepaper explicitly distinguishes the two.

---

## Table of Contents

- [Motivation](#motivation)
- [Abstract](#abstract)
- [1. Problem Statement](#1-problem-statement)
- [2. Core Thesis](#2-core-thesis)
- [3. Scope and Non-Goals](#3-scope-and-non-goals)
  - [In Scope](#in-scope)
  - [Out of Scope](#out-of-scope)
- [4. Architectural Overview](#4-architectural-overview)
  - [4.1 Measurement and Control](#41-measurement-and-control)
  - [4.2 Adaptive Routing](#42-adaptive-routing)
  - [4.3 Conditional Depth and Specialization](#43-conditional-depth-and-specialization)
  - [4.4 Offline Optimization Pipeline](#44-offline-optimization-pipeline)
- [5. Scientific Positioning](#5-scientific-positioning)
- [6. Formal Summary](#6-formal-summary)
- [7. Feasibility Summary](#7-feasibility-summary)
- [8. Experimental Program](#8-experimental-program)
  - [Stage 1: Instrumentation and Baseline](#stage-1-instrumentation-and-baseline)
  - [Stage 2: Adaptive Routing](#stage-2-adaptive-routing)
  - [Stage 3: Conditional Depth and Compression](#stage-3-conditional-depth-and-compression)
  - [Stage 4: Closed-Loop Offline Adaptation](#stage-4-closed-loop-offline-adaptation)
  - [Stage 5: Custom Hardware Research](#stage-5-custom-hardware-research)
- [9. Documentation Guide](#9-documentation-guide)
- [10. Selected Prior Work](#10-selected-prior-work)
- [11. Closing Position](#11-closing-position)
- [License](#license)

---

## 1. Problem Statement

Let a deployed model serve a workload distribution $\mathcal{D}$ over inputs $x$ with desired outputs $y$. In production, the operator cares about more than task loss. The system must also satisfy budgets over:

- expected energy per request or per generated token
- latency and tail latency
- memory bandwidth pressure
- thermal safety margins and throttling frequency
- throughput under realistic traffic conditions

Current frontier systems often optimize quality first and apply deployment optimizations later. TBI starts from a narrower and more testable premise:

> **Machine-learning systems should be designed and updated to operate under explicit physical budgets, not merely optimized for task quality in isolation.**

This is a systems claim, not a metaphysical claim about intelligence.

---

## 2. Core Thesis

TBI advances four claims.

1. **Deployment cost should be modeled during optimization.**  
   The relevant deployment costs are typically energy, latency, memory traffic, and thermal risk. These costs may enter the optimization process through measured statistics, learned surrogates, or constrained architecture search.

2. **Conditional computation is a primary mechanism for efficiency.**  
   Many workloads are heterogeneous. Some requests can be handled by caches, retrieval, smaller models, or shallow computation, while others require deeper or specialized paths. Routing is therefore central.

3. **Telemetry should inform periodic model and policy updates.**  
   Runtime traces can identify consistently expensive paths, poorly calibrated routers, and domains that benefit from distillation or compression.

4. **Near-term and long-term claims must be separated.**  
   Adaptive routing, early exit, MoE-style specialization, compression, and offline retraining are feasible today. Fine-grained hot, warm, and cold parameter residency across SRAM, HBM, and non-volatile storage is a future hardware research direction.

---

## 3. Scope and Non-Goals

### In Scope

- telemetry-informed adaptive inference
- routing across multiple fidelity levels
- early exit and conditional depth
- expert or domain specialization
- offline compression, pruning, quantization, and distillation
- constrained optimization using learned cost surrogates
- measurable evaluation under power, latency, and throughput budgets

### Out of Scope

- a claim that physical thermodynamic variables are exactly differentiable during ordinary training
- a claim that current GPUs expose fine-grained SRAM residency control for multi-gigabyte parameter blocks
- a recommendation to mutate production weights in place without offline evaluation, canaries, and rollback
- a new general theory of intelligence

---

## 4. Architectural Overview

TBI organizes deployment into four interacting layers.

### 4.1 Measurement and Control

A telemetry layer records power, energy proxies, latency, memory pressure, clock state, throttle indicators, and thermal headroom. These signals are used for diagnosis and control. Temperature is treated primarily as a **state and safety variable**, not as a precise per-request optimization target.

### 4.2 Adaptive Routing

A lightweight gate decides whether a request should be served by:

- a cache or retrieval answer
- a small model
- a shallow path through a larger model
- a full model path
- a specialized expert or domain model

The gate may use prompt features, embeddings, recent telemetry state, calibration statistics, and retrieval signals.

### 4.3 Conditional Depth and Specialization

Within the heavy path, the system may still save work through:

- early exit at calibrated checkpoints
- expert routing
- domain specialization
- precision control
- batching and scheduling policies that adapt to current system state

### 4.4 Offline Optimization Pipeline

Deployment traces are used to generate new candidates:

- pruned or quantized variants
- distilled smaller models
- recalibrated routing policies
- improved retrieval or cache indexes

Promotion into production occurs only after offline evaluation and staged rollout.

---

## 5. Scientific Positioning

TBI is best understood as a synthesis of ideas already present in the literature:

- scaling-law thinking from large-model training
- conditional computation and mixture-of-experts routing
- early-exit networks and adaptive computation
- model cascades and selective prediction
- pruning, quantization, and distillation
- telemetry-aware systems control

The novelty of TBI is therefore **integrative** rather than foundational. Its scientific value depends on whether this integration produces materially better quality-per-joule, quality-per-second, and throughput-per-watt on real workloads.

---

## 6. Formal Summary

A high-level TBI objective can be written as a constrained or relaxed optimization problem:

$$
\min \; \mathbb{E}\left[\mathcal{L}_{\text{task}}\right]
\quad
\text{subject to}
\quad
\mathbb{E}[J] \leq B_J,\;
\mathbb{E}[L] \leq B_L,\;
\mathbb{E}[M] \leq B_M,\;
\Pr(T > T_{\text{safe}}) \leq \delta
$$

or, in Lagrangian form,

$$
\min \; \mathbb{E}\left[
\mathcal{L}_{\text{task}}
+ \lambda_J J
+ \lambda_L L
+ \lambda_M M
+ \lambda_\chi \chi
\right]
$$

where:

- $J$ is energy cost
- $L$ is latency cost
- $M$ is memory or bandwidth cost
- $\chi$ is a safety or throttling penalty
- $T$ is thermal state
- the costs are estimated from measurement or learned surrogates

In TBI, these costs are not assumed to be analytically differentiable with respect to all model weights. Instead, they are estimated and optimized over the **controllable decisions**: route choice, depth, expert selection, precision, compression level, and architecture choice.

Detailed formalism appears in [docs/01_mathematical_foundations.md](docs/01_mathematical_foundations.md).

---

## 7. Feasibility Summary

| Component | Near-Term Feasibility | Interpretation |
| --- | --- | --- |
| Telemetry collection on commodity accelerators | High | Practical today, with coarse time resolution and vendor-specific limitations |
| Telemetry-aware routing and threshold control | High | Practical today if calibrated conservatively |
| Early exit and conditional depth | Medium to High | Practical, but highly workload-dependent and sensitive to miscalibration |
| Domain or expert specialization | High | Already supported by multiple serving patterns |
| Surrogate-based cost-aware optimization | Medium | Feasible, but requires careful trace collection and validation |
| Exact per-request thermal attribution | Low | Usually too noisy or too slow for precise credit assignment |
| Fine-grained hot, warm, cold parameter residency in SRAM and HBM | Low on current hardware | Best treated as future custom-hardware work |
| In-place autonomous self-modification of production models | Low and unsafe | Should be replaced by an offline optimization and staged rollout pipeline |

---

## 8. Experimental Program

A credible TBI program should be staged.

### Stage 1: Instrumentation and Baseline

- collect energy, latency, throughput, and memory-pressure traces
- identify the most expensive routes and workload slices
- establish a strong monolithic baseline

### Stage 2: Adaptive Routing

- train and calibrate a lightweight gate
- evaluate cost-quality trade-offs under conservative escalation rules
- report risk-coverage and cost-coverage curves

### Stage 3: Conditional Depth and Compression

- add early exit checkpoints or specialized experts
- train compressed student models
- compare quality-per-joule and latency distributions against the baseline

### Stage 4: Closed-Loop Offline Adaptation

- retrain routing and compression policies from production traces
- validate candidates offline
- promote only after canary evaluation

### Stage 5: Custom Hardware Research

- explore whether finer-grained parameter residency control changes the operating frontier
- treat these experiments as separate from claims about commodity accelerators

---

## 9. Documentation Guide

- [docs/01_mathematical_foundations.md](docs/01_mathematical_foundations.md)  
  Formal problem statement, constrained optimization, surrogate cost modeling, and limitations.

- [docs/02_tier1_telemetry.md](docs/02_tier1_telemetry.md)  
  Measurement model, metric hierarchy, attribution limits, and control policy.

- [docs/03_tier2_routing.md](docs/03_tier2_routing.md)  
  Adaptive routing, gate calibration, escalation logic, and evaluation methodology.

- [docs/04_tier3_sparse_workers.md](docs/04_tier3_sparse_workers.md)  
  Conditional depth, specialization, batching interactions, and the boundary between current practice and future hardware.

- [docs/05_background_daemons.md](docs/05_background_daemons.md)  
  Offline optimization pipeline, compression workflow, evaluation gates, and deployment safety.

---

## 10. Selected Prior Work

TBI draws on several established lines of research, including:

- scaling laws for large language models
- compute-optimal training
- mixture-of-experts and conditional computation
- adaptive computation time and early-exit networks
- model distillation
- pruning and quantization
- hardware-aware model optimization and systems profiling

Representative examples include work by Kaplan et al., Hoffmann et al., Shazeer et al., Graves, Teerapittayanon et al., and Hinton et al.

---

## 11. Closing Position

TBI is proposed as a research agenda for **quality under physical constraint**. Its strongest near-term form is not literal thermodynamic backpropagation, but a disciplined closed loop connecting telemetry, routing, compression, and periodic retraining. The central hypothesis is empirical: can this loop move the Pareto frontier of quality, cost, and reliability on real workloads?

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

---

_tbi-spec — Thermodynamically Bounded Intelligence Open Technical Specification_
