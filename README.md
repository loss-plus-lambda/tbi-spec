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
  - [4.5 Self-Optimization Lifecycle](#45-self-optimization-lifecycle)
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

TBI is formally characterized by four properties:

**Property I — Hardware-Aware Optimization:**  
The core optimization objective explicitly incorporates real-time physical hardware cost terms, penalizing execution pathways that generate excessive thermal load, FLOP expenditure, or memory bandwidth pressure relative to the marginal improvement in task performance they provide. Hardware cost is a first-order term in the optimization objective, not merely a post-hoc profile.

**Property II — Conditional Runtime Sparsity:**  
Parameter execution is not monolithic. Worker parameter blocks are maintained in low-power standby states by default and dynamically provisioned into active execution only when routed input complexity demands their engagement. The active computational footprint at any instant is proportional to the informational complexity of the work being resolved.

**Property III — Closed-Loop Telemetry Integration:**  
Physical hardware telemetry collected during production inference—junction temperatures, FLOP overhead per execution path, memory bandwidth saturation events, DVFS throttle triggers—continuously feeds back into the structural optimization objective. The system evolves lighter execution pathways for context domains that have historically caused thermodynamic bottlenecks.

**Property IV — Autonomous Background Maintenance:**  
During periods of reduced external query load, the system autonomously executes maintenance daemons that prune redundant parameters, consolidate episodic memory indices, and distill knowledge into more compact structural representations. This self-optimization lifecycle operates entirely within the thermal and power envelopes reported by the hardware telemetry substrate.

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

TBI organizes deployment into four interacting layers. The diagram below maps the primary data-flow paths, control signals, and feedback loops between all architectural components.

```mermaid
%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 48, "rankSpacing": 72}}}%%
flowchart TB
    classDef tier1 fill:#0d3349,stroke:#4db8d4,stroke-width:2px,color:#cce7f5
    classDef tier2 fill:#1a0d33,stroke:#9b59b6,stroke-width:2px,color:#e8d5ff
    classDef tier3 fill:#0d3318,stroke:#27ae60,stroke-width:2px,color:#d5ffe5
    classDef fp    fill:#332200,stroke:#f39c12,stroke-width:2px,color:#fff3cc
    classDef dl    fill:#330d0d,stroke:#e74c3c,stroke-width:2px,color:#ffd5d5
    classDef io    fill:#1a1a2a,stroke:#95a5a6,stroke-width:2px,color:#ffffff

    INPUT(["Context Input<br/>Inbound API Stream"]):::io

    subgraph T2["TIER 2 — Asynchronous Routing & Gating Bus"]
        direction LR
        EA["Entropy Assessor<br/>(Shallow Gate / Proxy Model)"]:::tier2
        ESM["Execution State Manager<br/>(Routing Table / FSM)"]:::tier2
        EA --> ESM
    end

    INPUT --> EA

    subgraph FP["Static / Compiled Fast Path"]
        FPD["Lookup Tables · Cached Fields<br/>Ultra-Light Models · KV Cache"]:::fp
    end

    subgraph T3["TIER 3 — Sparse Worker Blocks"]
        T3D["Dynamic Activation · Layer-Gating / Early-Exit<br/>On-Chip Worker Pools · SRAM Mapping"]:::tier3
    end

    ESM -- "Low-Entropy Path" --> FPD
    ESM -- "High-Entropy Trigger" --> T3D

    subgraph T1["TIER 1 — Telemetry & Bare-Metal Substrate Layer"]
        direction LR
        HIB["Hardware Interface Bus<br/>(NVML / DCGM / ASIC Counters)"]:::tier1
        HCU["Homeostatic Control Unit<br/>(DVFS Loop / Power States)"]:::tier1
        HIB --> HCU
    end

    FPD -- "Execution Results + FLOP + Telemetry" --> HIB
    T3D -- "Execution Results + FLOP + Telemetry" --> HIB

    subgraph DL["Telemetry Feedback Data Lake"]
        DLD["Bottleneck Tagging Engine<br/>Vector λ⃗ Adjuster · Next Optimization Input"]:::dl
    end

    HCU -- "T_vec = [τ_j, P_draw, M_util, ω_clock] · sub-ms" --> DLD
    HCU -. "T_vec thermal feedback (async)" .-> EA
    DLD --> OPT(["Training Optimization Phase"]):::io
```

> **Note:** Dashed arrows represent background asynchronous signals. The T_vec feedback path from Tier 1 to the Data Lake is a non-blocking async write stream and does not gate the primary inference path.

---

### 4.1 Measurement and Control

The telemetry layer maintains a continuous state vector $\vec{T}$ representing the physical operational state of every execution node:

$$\vec{T} = [\tau_{\text{junction}},\; P_{\text{draw}},\; M_{\text{util}},\; \omega_{\text{clock}}]$$

**Dual-Window Telemetry Architecture.** These signals operate on fundamentally incompatible time scales and must not be collapsed into a single polling loop. TBI enforces a strict two-window separation:

- **Fast window** ($\leq 1$ ms): kernel-level FLOP counters, request timestamps, achieved memory bandwidth, and route-level latency. These are gathered at kernel or micro-batch granularity with zero sensor lag and form the primary optimization substrate.
- **Slow window** ($\geq 5$–$50$ ms): junction temperature, DVFS state, board power, and thermal headroom. These are consumed exclusively as control and safety signals, not as per-request cost targets.

Conflating the two windows — for example, attributing a vendor-reported 10 ms power average to a 3 ms token generation event — produces systematic attribution errors that corrupt both surrogate cost models and routing policies. The separation is a hard architectural requirement.

**Analytical Energy Proxy.** Raw integration of vendor power readings ($\int P(t)\,dt$) is replaced by a FLOP-count proxy calibrated from periodic offline profiling:

$$\hat{J}_k = \alpha_k \cdot \text{FLOP}(x,\, k) + \beta_k$$

where $\alpha_k$ is a per-route joules-per-FLOP coefficient and $\beta_k$ is a per-route static memory-access cost term, both estimated from micro-benchmarks at representative batch shapes. This proxy is deterministic, zero-latency, and immune to NVML polling jitter and idle-power baseline drift.

**Sensor-Delay Compensation.** The routing gate at request time consumes $\vec{T}(t - \Delta t_{\text{sensor}})$, a delayed observation with $\Delta t_{\text{sensor}} \in [1, 10]$ ms on commodity accelerators. TBI models this delay explicitly. Threshold updates from the Homeostatic Control Unit are passed through a low-pass filter with cutoff frequency:

$$\omega_c \leq \frac{1}{2\,\Delta t_{\text{sensor}}}$$

The DVFS adjustment controller gain $K$ is bounded by the Nyquist stability condition applied to the delay-augmented plant:

$$K \leq \frac{\pi}{2\,\omega_c\,\Delta t_{\text{sensor}}}$$

Gain selections exceeding this bound risk sawtooth oscillation in the DVFS state across the load window and must not be deployed. Temperature remains a **state and safety variable** — the HCU uses it only to trigger conservative, reversible threshold adjustments within prevalidated bounds.

→ Full specification: [docs/02_tier1_telemetry.md](docs/02_tier1_telemetry.md)

### 4.2 Adaptive Routing

An **Entropy Assessor** — a deliberately shallow, computationally inexpensive gating sub-network or heuristic proxy — evaluates the informational complexity of each inbound context vector before any heavy computation is engaged. Based on this assessment, it routes to one of two execution paths:

- **Fast Path (Low-Entropy):** Routine, low-variance inputs are resolved using compiled lookup tables, cached activation fields, or ultra-lightweight models at near-zero energy expenditure.
- **Slow Path (High-Entropy):** Complex, high-variance inputs trigger provisioning of the appropriate deep worker block clusters.

The gate may use prompt features, embeddings, recent telemetry state, calibration statistics, and retrieval signals.

**Formal Gate Efficiency Constraint.** The Entropy Assessor is thermodynamically justified only when its execution overhead is bounded by the savings it enables. Let $C_g$ denote the total gate cost in combined FLOPs and memory bandwidth, $C_f$ the fast-path cost, $C_s$ the slow-path cost, and $p^*$ the expected escalation rate on the target workload. Two independent constraints must both hold:

$$C_g \leq \epsilon_f \cdot C_f, \qquad \epsilon_f \in [0.01,\; 0.05]$$

$$C_g \leq \epsilon_n \cdot (1 - p^*)(C_s - C_f), \qquad \epsilon_n \leq 0.10$$

The first constraint — gate overhead at most 1–5% of the fast path's own FLOP and bandwidth budget — prevents the gate from materially degrading the fast path's energy advantage. The second ensures the gate consumes at most 10% of the routing savings margin, preserving a 10× efficiency buffer against measurement error and workload variation. Any gate architecture that violates either constraint at the observed $p^*$ must be rejected or restructured before deployment.

**Shared Embedding Architecture.** If the gate uses a compact encoder to derive request embeddings, that encoder is promoted to a **shared prefill step** executed once per request — its cost is amortized across downstream routing, retrieval, and KV-cache operations and is not counted within the per-decision gate budget $C_g$. Encoder weights must remain as persistent device-memory residents to avoid cold-load latency. The gate's real-time decision logic — threshold evaluation, FSM lookup, telemetry-aware adjustment — must independently satisfy both constraints above at $\leq 1$–$5\%$ of the fast path's combined FLOP and memory-bandwidth budget.

**Coverage Requirement.** Routing gains are highly workload-dependent. Any evaluation must report the risk-coverage curve across the full empirical range of escalation rates $p \in [0, 1]$. The gate must demonstrate positive thermodynamic ROI at the 90th-percentile $p$ value observed in production traffic. Efficiency claims reported only at the median operating point are insufficient for publication-grade evaluation.

→ Full specification: [docs/03_tier2_routing.md](docs/03_tier2_routing.md)

### 4.3 Conditional Depth and Specialization

Within the heavy path, the system may still save work through:

- early exit at calibrated checkpoints
- expert routing
- domain specialization
- precision control
- batching and scheduling policies that adapt to current system state

**Layer-gating mechanics** allow inference to issue an early-exit return at intermediate layer $N$ when a high-confidence prediction threshold has been met, bypassing the computational cost of all subsequent layers—potentially eliminating a substantial fraction of the total FLOP budget for a given token prediction.

→ Full specification: [docs/04_tier3_sparse_workers.md](docs/04_tier3_sparse_workers.md)

### 4.4 Offline Optimization Pipeline

Deployment traces are used to generate new candidates:

- pruned or quantized variants
- distilled smaller models
- recalibrated routing policies
- improved retrieval or cache indexes

Promotion into production occurs only after offline evaluation and staged rollout.

### 4.5 Self-Optimization Lifecycle

When external query traffic falls below a configurable threshold, or during designated maintenance windows, TBI transitions available compute into **background optimization daemons**. These daemons execute within the physical constraints reported by the Tier 1 telemetry substrate. Three primary daemon classes are defined:

| Daemon Class | Function | Mechanism |
| --- | --- | --- |
| **Sparsity Daemons** | Eliminate redundant parameters | Magnitude and gradient-based pruning passes; heavy quantization of low-contribution weights |
| **Memory Index Consolidation (Samskaras)** | Compress episodic inference logs | Background merge of transactional records into high-dimensional relational vector anchors |
| **Recursive Knowledge Distillation** | Compact the model's own parameter density | Automated self-play loops training a sub-model to mirror heavy cluster outputs under a constrained FLOP budget |

**Net-Negative Joule Amortization Constraint.** No daemon class is thermodynamically justified by its intent alone. Every candidate promotion must pass a mandatory energy amortization gate before entering the staging pipeline:

$$\frac{E_{\text{daemon}}}{|\Delta J_{\text{serving}}| \cdot R_{\text{projected}}} \leq H_{\text{amortize}}$$

where $E_{\text{daemon}}$ is the measured energy cost of the daemon run, $|\Delta J_{\text{serving}}|$ is the measured per-request energy improvement under the new candidate, $R_{\text{projected}}$ is the expected request count over the model's remaining deployment window, and $H_{\text{amortize}}$ is a configurable maximum amortization horizon (default: 30 days). A candidate that passes quality and latency gates but fails this inequality is a net energy consumer over its deployment lifetime and must not be promoted.

**Diminishing-Returns Stopping Criterion.** Iterative daemon passes exhibit concave marginal returns: successive pruning or compression rounds recover less redundancy at constant compute cost. Daemon cycles terminate automatically when marginal per-request energy improvement falls below an operator-defined threshold $\kappa$:

$$\frac{\partial\,|\Delta J_{\text{saving}}|}{\partial\,n_{\text{rounds}}} \leq \kappa$$

Running daemon iterations past this boundary inverts the energy ROI and is treated as a configuration defect.

**Grid-Aware Efficiency Policy.** TBI's thermodynamic framing requires that energy accounting not conflate Joule consumption with carbon impact. Off-peak scheduling that minimizes GPU idle time can worsen grid carbon intensity when off-peak hours are served by baseload carbon generation rather than renewable dispatch. TBI therefore mandates a **Grid-Aware Efficiency Policy** for daemon scheduling:

- Daemon windows are scheduled against a real-time grid carbon-intensity forecast (e.g., WattTime or Electricity Maps). Daemon execution is preferred when marginal grid carbon intensity $\gamma(t)$ is at or below the rolling 24-hour median $\bar{\gamma}$.
- The effective thermodynamic scheduling metric is the carbon-adjusted energy cost $E_{\text{carbon}} = E_{\text{daemon}} \cdot \gamma(t)$, not raw Joules alone.

**Thermal Recovery Gap.** Daemon runs at full GPU utilization delay thermal recovery of the package. A mandatory cooldown gap $\Delta t_{\text{cool}}$ is enforced between the end of any daemon window and the start of the subsequent peak traffic window:

$$\Delta t_{\text{cool}} \geq R_\theta C_\theta \ln\!\left(\frac{\tau_{\text{daemon}} - \tau_{\text{ambient}}}{\tau_{\text{target}} - \tau_{\text{ambient}}}\right)$$

where $R_\theta$ is the package thermal resistance, $C_\theta$ is thermal capacitance, $\tau_{\text{daemon}}$ is the measured junction temperature at daemon completion, and $\tau_{\text{target}}$ is the peak-traffic thermal headroom target. Where hardware thermal models are unavailable, this window defaults to empirical measurement across at least 20 run-cooldown cycles on the target hardware class.

The combined effect is a self-optimization lifecycle that is verifiably net-positive over each amortization horizon — bounded by both the hardware envelope and the carbon envelope of the grid it inhabits.

→ Full specification: [docs/05_background_daemons.md](docs/05_background_daemons.md)

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

A high-level TBI objective is stated as a constrained optimization problem over the serving system's controllable decisions:

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

**Continuous Relaxation for Discrete Routing.** Route selection, early-exit depth, expert assignment, and quantization level are inherently discrete decisions. The standard Lagrangian relaxation of a mixed-integer program does not generally yield integer-feasible solutions; the duality gap can be nonzero and problem-dependent. TBI therefore adopts a **stochastic policy formulation** for routing:

$$k \sim \pi_\phi(x, s) = \text{Gumbel-Softmax}\!\left(g_\phi(x, s),\; \tau_{\text{temp}}\right)$$

where $g_\phi(x, s)$ are the gate logits, $\tau_{\text{temp}}$ is a temperature parameter annealed during offline policy optimization, and the Gumbel-Softmax reparameterization provides differentiable surrogate gradients through the discrete routing decision. At deployment, the policy is evaluated in hard mode (argmax). For multi-step escalation trees where Gumbel-Softmax is structurally inapplicable, the REINFORCE estimator with a learned variance-reducing baseline is used instead.

The Lagrangian relaxation is applied to the **continuous** policy parameters and surrogate cost terms:

$$
\min_{\theta, \phi, a} \;\mathbb{E}\!\left[
\mathcal{L}_{\text{task}}
+ \lambda_J J
+ \lambda_L L
+ \lambda_M M
+ \lambda_\chi \chi
\right]
$$

where:

- $J$ is energy cost, estimated via the FLOP-count analytical proxy $\hat{J}_k = \alpha_k \cdot \text{FLOP}(x,k) + \beta_k$
- $L$ is latency cost
- $M$ is memory or bandwidth cost
- $\chi$ is a safety or throttling penalty
- $T$ is thermal state, treated as a constraint variable and never differentiated directly
- all costs are estimated from measurement or learned surrogates

**Stable Lambda Update with PID Damping.** Naive proportional dual ascent produces limit cycles when cost feedback carries delay $\Delta t$. TBI mandates a derivative-damped update with hysteresis:

$$
\lambda_k^{(t+1)} = \max\!\left(0,\; \lambda_k^{(t)} + \alpha_\lambda \cdot e_k^{(t)} + \mu \cdot \frac{e_k^{(t)} - e_k^{(t-1)}}{\Delta t} \right)
$$

where $e_k^{(t)} = \hat{C}_k^{(t)} - B_k$ is the constraint violation measured as an exponentially-weighted moving average over a window of $W$ requests, $\alpha_\lambda$ is the proportional gain, and $\mu$ is the derivative damping coefficient. The proportional gain satisfies:

$$\alpha_\lambda \leq \frac{1}{L_\psi \cdot W}$$

where $L_\psi$ is the Lipschitz constant of the surrogate cost model $\hat{\mathbf{c}}_\psi$. A hysteresis band of width $\pm\eta_k$ around each constraint boundary $B_k$ prevents rapid multiplier cycling under borderline violation conditions: the multiplier updates only when $|e_k^{(t)}| > \eta_k$.

**Importance-Weighted Surrogate Retraining.** The surrogate $\hat{\mathbf{c}}_\psi$ is trained on traces collected under a prior policy $\pi_\phi^{\text{old}}$. When the routing policy is updated to $\pi_\phi^{\text{new}}$, covariate shift invalidates the surrogate on the new request distribution. Surrogate retraining is triggered whenever:

$$\|\phi^{\text{new}} - \phi^{\text{old}}\|_2 > \delta_\phi$$

Retraining uses importance-weighted empirical risk minimization:

$$\min_\psi \sum_i w_i \left\|\hat{\mathbf{c}}_\psi(z_i) - c_i\right\|_2^2 + \eta\|\psi\|_2^2, \qquad w_i = \frac{p_{\text{new}}(z_i)}{p_{\text{old}}(z_i)}$$

where the density ratio $w_i$ is estimated from a lightweight discriminator trained on the old and new trace distributions. This correction prevents the surrogate from silently degrading the routing policy after each optimization cycle.

In TBI, physical costs are not assumed to be analytically differentiable with respect to all model weights. Optimization operates over the **controllable decisions** — route choice, depth, expert selection, precision, compression level, and architecture choice — with the mechanisms above ensuring the optimization loop is stable, self-correcting, and free of duality-gap pathologies on discrete action spaces.

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
