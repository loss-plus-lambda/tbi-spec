# TBI Whitepaper Appendix C: Adaptive Routing and Selective Computation

> **Document:** `docs/03_tier2_routing.md`  
> **Depends on:** [docs/01_mathematical_foundations.md](01_mathematical_foundations.md), [docs/02_tier1_telemetry.md](02_tier1_telemetry.md)  
> **Related:** [docs/04_tier3_sparse_workers.md](04_tier3_sparse_workers.md)

---

## Overview

This appendix defines the routing layer of TBI. The routing problem is simple to state: for each request, choose the cheapest execution path that preserves acceptable quality. This is a familiar idea in model cascades, selective prediction, and conditional computation. TBI places it at the center of the deployment architecture.

The strongest version of the routing claim is workload-dependent. Adaptive routing is most valuable when the request distribution is heterogeneous, repetitive, or domain-skewed. It is less valuable when nearly every request already requires the largest model path.

---

## 1. Routing as a Decision Problem

Let the set of available paths be:

$$
\mathcal{K} =
\{
k_{\text{cache}},
k_{\text{small}},
k_{\text{shallow}},
k_{\text{full}},
k_{\text{expert}}
\}
$$

Not every deployment needs all paths, but a practical TBI system should expose at least two or three fidelity levels.

For input $x$ and runtime state $s$, the gate selects:

$$
k = \pi_\phi(x, s)
$$

The selected path yields a prediction $\hat{y}_k$ and an expected cost $\hat{c}_k$.

The routing objective is to minimize expected cost while keeping quality loss within an acceptable bound.

---

## 2. Inputs to the Gate

A realistic gate should combine features from four sources.

### 2.1 Request Features

- prompt length
- token statistics
- domain markers
- retrieval hits or cache matches
- structural features such as code versus prose

### 2.2 Embedding Features

A compact encoder can provide an embedding of the request for:

- novelty estimation
- domain prediction
- proximity to previously seen prompts
- confidence features derived from a small model

**Shared Prefill Architecture.** When a compact encoder is used, it must be promoted to a **shared prefill step** executed once per request — its FLOP and bandwidth cost is amortized across downstream routing, retrieval, and KV-cache operations and is not counted within the per-decision gate budget $C_g$. Encoder weights must remain as persistent device-memory residents to avoid cold-load latency on the critical inference path. The gate's real-time decision logic must independently satisfy the gate efficiency constraints in §8.4 without relying on the encoder's separate amortization.

### 2.3 System-State Features

The gate may use current telemetry or queue state such as:

- recent latency
- queue depth
- memory pressure
- throttle flags
- thermal headroom

These features help the gate respect deployment budgets.

### 2.4 Calibration Features

The gate should also use self-estimated uncertainty or confidence. A route that looks cheap but uncertain should escalate.

---

## 3. Conservative Routing Principle

A strong TBI routing policy follows one principle:

> **Cheap paths may answer only when calibrated confidence is sufficient; otherwise the system escalates.**

This turns routing into a selective prediction problem rather than a reckless approximation scheme.

A generic confidence rule is:

$$
\text{commit}(x, k) =
\mathbb{1}
\left[
u_\phi(x, s, k) \leq \tau_k
\right]
$$

where:

- $u_\phi$ is an uncertainty estimate
- $\tau_k$ is a calibrated threshold for path $k$

If the uncertainty is too high, the request is sent to a more capable path.

---

## 4. Cost-Aware Routing Objective

A route-aware objective can be written as:

$$
\min_{\phi}
\;\mathbb{E}
\left[
\mathcal{L}_{\text{task}}(f_{\theta, \pi_\phi(x, s)}(x), y)
+ \lambda^\top \hat{\mathbf{c}}_\psi(x, \pi_\phi(x, s), s)
\right]
$$

In practice, this is often implemented through:

- policy learning from teacher labels
- bandit-style route comparison
- imitation of the cheapest acceptable path
- threshold calibration on held-out data

The important scientific point is that the routing decision must be trained and evaluated under **quality and cost together**, not under cost alone.

---

## 5. Suggested Path Hierarchy

A publication-grade TBI design should describe routes in realistic terms.

### 5.1 Cache or Retrieval Route

This path returns an answer only when there is a strong cache hit, retrieval hit, or known template match. It is the cheapest path and the safest when similarity is high.

### 5.2 Small-Model Route

A small model can handle many routine or templated cases. This route is practical today and often provides the highest engineering return on investment.

### 5.3 Shallow Route

A larger model can be run only to a shallow checkpoint and exit early if confidence is sufficient. This route provides a middle ground between a small standalone model and the full heavy path.

### 5.4 Full Route

This is the baseline path that preserves maximum quality.

### 5.5 Expert Route

For deployments with clear domain structure, a specialized expert or domain model may outperform the generic full route on quality-per-cost.

---

## 6. Calibration and Evaluation

Routing should be judged by more than average cost reduction.

### 6.1 Core Metrics

- quality relative to the full-route baseline
- expected cost reduction
- p95 and p99 latency
- route coverage
- regret versus the full route
- escalation frequency
- calibration error of uncertainty estimates
- thermodynamic ROI of the gate at the **90th-percentile escalation rate** $p_{90}$ observed in production traffic; positive ROI must be demonstrated across the full empirical range $p \in [0, 1]$, not only at the median operating point

### 6.2 Risk-Coverage View

A useful evaluation is the risk-coverage curve. As the system commits more requests to cheap routes, quality usually degrades. The gate is good if it pushes that frontier outward.

### 6.3 Workload Dependence

Strong routing gains are most plausible on:

- repetitive enterprise traffic
- narrow domain assistants
- retrieval-heavy workloads
- mixed traffic with a large routine component

Claims about very large gains on open-ended general-assistant workloads should be treated as hypotheses requiring evidence, not as assumptions.

---

## 7. Telemetry-Aware Routing

TBI allows routing thresholds to depend on current deployment state:

$$
\tau_k(t) = \tau_{k,0} + \Delta\tau_k(s_t)
$$

This does not mean the system should blindly degrade quality whenever temperature rises. The safer interpretation is:

- under pressure, bias toward cheaper paths only within calibrated error limits
- preserve a hard escalation path when uncertainty is high
- use telemetry as a budget signal, not as a replacement for quality calibration

---

## 8. Failure Modes

### 8.1 Overconfident Cheap Paths

A cheap route that is poorly calibrated can fail silently. This is the most dangerous routing error.

### 8.2 Shifted Workloads

If the workload changes, the route distribution can drift and invalidate calibration.

### 8.3 Telemetry-Induced Misrouting

If telemetry signals are noisy or lagged, thresholds may overreact or underreact.

### 8.4 Gate Cost Creep

A routing stack that grows too large defeats its own purpose. The gate's thermodynamic justification requires two independent constraints to hold simultaneously. Let $C_g$ denote the total gate cost in combined FLOPs and memory bandwidth, $C_f$ the fast-path cost, $C_s$ the slow-path cost, and $p^*$ the expected escalation rate on the target workload:

$$C_g \leq \epsilon_f \cdot C_f, \qquad \epsilon_f \in [0.01,\; 0.05]$$

$$C_g \leq \epsilon_n \cdot (1 - p^*)(C_s - C_f), \qquad \epsilon_n \leq 0.10$$

The first constraint limits the gate to at most 1–5% of the fast path's own FLOP and bandwidth budget, preventing it from materially degrading the fast path's energy advantage. The second ensures the gate consumes at most 10% of the routing savings margin, preserving a 10× efficiency buffer against measurement error and workload variation. Both constraints must hold at the observed $p^*$, not just at the median. Any gate architecture that violates either constraint must be rejected or restructured before deployment.

---

## 9. Deployment Recommendations

For current production systems, a defensible routing rollout is:

1. start with two routes: small path and full path
2. add retrieval or cache answers where precision is high
3. add shallow-path early exit only after quality calibration
4. evaluate with strict escalation and rollback
5. introduce domain experts only when the workload justifies them

This staged approach is more credible than assuming a highly complex routing fabric from the start.

---

## 10. Summary

The TBI routing layer is one of the most feasible parts of the overall proposal. Its scientific success depends on calibration, escalation discipline, and workload realism. The strongest claim is not that most requests will always take a cheap path, but that a well-calibrated gate can move the quality-cost frontier on workloads with meaningful heterogeneity.
