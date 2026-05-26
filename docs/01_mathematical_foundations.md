# TBI Whitepaper Appendix A: Mathematical Foundations

> **Document:** `docs/01_mathematical_foundations.md`  
> **Depends on:** [README.md](../README.md)  
> **Related:** [docs/02_tier1_telemetry.md](02_tier1_telemetry.md), [docs/03_tier2_routing.md](03_tier2_routing.md)

---

## Overview

This appendix formalizes Thermodynamically Bounded Intelligence as a problem of **constrained adaptive inference** and **surrogate-guided optimization**. The goal is not to claim that thermal sensors, board power, or memory controller behavior are exactly differentiable with respect to arbitrary model weights during standard training. The goal is narrower and more defensible: deployment cost can be modeled, predicted, and incorporated into optimization over the controllable decisions of a serving system.

Those controllable decisions include route choice, early-exit thresholds, expert selection, quantization level, architecture choice, and model compression level. In this setting, hardware cost is not an afterthought; it is part of the objective.

---

## 1. From Task-Only Optimization to Deployment-Aware Optimization

### 1.1 Standard Objective

A standard supervised or language-model objective can be written as:

$$
\min_{\theta} \;
\mathbb{E}_{(x, y) \sim \mathcal{D}}
\left[
\mathcal{L}_{\text{task}}(f_\theta(x), y)
\right]
$$

This objective is successful because it directly optimizes prediction quality. However, it does not explicitly optimize the downstream cost of serving the model under realistic deployment constraints.

### 1.2 Deployment Costs of Interest

For deployment, the operator typically cares about a vector of costs rather than a single scalar:

$$
\mathbf{c} =
\begin{bmatrix}
J \\
L \\
M \\
\chi
\end{bmatrix}
$$

where:

- $J$ is energy cost, such as joules per request or joules per generated token
- $L$ is latency cost, such as mean or tail latency
- $M$ is memory or bandwidth cost
- $\chi$ is a safety penalty, such as throttle events or violation indicators

Temperature $T$ remains important, but it is most scientifically useful as a **state variable** or **constraint variable**, rather than as an exactly attributable per-request target.

---

## 2. Adaptive Inference Formulation

### 2.1 Decision Variables

Let:

- $\theta$ denote model parameters
- $\phi$ denote routing or gating parameters
- $a$ denote architectural or systems controls, such as checkpoint layout, expert configuration, or precision policy
- $s$ denote runtime state, including recent telemetry, queue state, and other control signals

Let the routing policy be:

$$
k = \pi_\phi(x, s)
$$

where $k$ indexes the chosen execution path. For example, $k$ may correspond to a cache answer, a small model, a shallow path, a full path, or a specialized expert.

The selected path produces an output:

$$
\hat{y} = f_{\theta, a, k}(x)
$$

### 2.2 Constrained Objective

A constrained formulation of TBI is:

$$
\min_{\theta, \phi, a}
\;
\mathbb{E}
\left[
\mathcal{L}_{\text{task}}(f_{\theta, a, \pi_\phi(x, s)}(x), y)
\right]
$$

subject to:

$$
\mathbb{E}[J] \leq B_J,
\qquad
\mathbb{E}[L] \leq B_L,
\qquad
\mathbb{E}[M] \leq B_M,
\qquad
\Pr(T > T_{\text{safe}}) \leq \delta
$$

where $B_J$, $B_L$, and $B_M$ are operator-defined budgets and $\delta$ is an acceptable thermal-risk threshold.

### 2.3 Lagrangian Relaxation

In practice, a relaxed objective is often easier to optimize:

$$
\min_{\theta, \phi, a}
\;
\mathbb{E}
\left[
\mathcal{L}_{\text{task}}
+ \lambda_J J
+ \lambda_L L
+ \lambda_M M
+ \lambda_\chi \chi
\right]
$$

This relaxation makes explicit that TBI is a **multi-objective optimization problem** over quality and deployment cost.

---

## 3. Surrogate Cost Models

### 3.1 Why Surrogates Are Necessary

A serving system depends on discrete route choices, scheduler behavior, batching interactions, memory hierarchy effects, and control loops such as DVFS. These phenomena are not cleanly differentiable with respect to all parameters of the network.

Therefore, TBI relies on **surrogate cost models** trained from deployment traces or targeted profiling runs.

### 3.2 Trace Dataset

Let a trace record be:

$$
\mathcal{T}_i = (z_i, c_i)
$$

where:

- $z_i = \Phi(x_i, k_i, s_i, a_i)$ is a feature vector describing the request, route, runtime state, and system configuration
- $c_i = [J_i, L_i, M_i, \chi_i]$ is the observed cost vector

A dataset of trace records is:

$$
\mathcal{D}_{\text{trace}} = \{(z_i, c_i)\}_{i=1}^{N}
$$

### 3.3 Surrogate Model

A learned cost model is:

$$
\hat{\mathbf{c}}_\psi(z) =
\begin{bmatrix}
\hat{J}_\psi(z) \\
\hat{L}_\psi(z) \\
\hat{M}_\psi(z) \\
\hat{\chi}_\psi(z)
\end{bmatrix}
$$

trained by minimizing:

$$
\min_{\psi}
\sum_{i=1}^{N}
\left\|
\hat{\mathbf{c}}_\psi(z_i) - c_i
\right\|_2^2
+ \eta \|\psi\|_2^2
$$

or, when appropriate, a mixture of regression and classification losses for continuous and discrete targets.

### 3.4 Use in Optimization

The deployment-aware objective then becomes:

$$
\min_{\theta, \phi, a}
\;
\mathbb{E}
\left[
\mathcal{L}_{\text{task}}
+ \lambda^\top \hat{\mathbf{c}}_\psi(\Phi(x, \pi_\phi(x, s), s, a))
\right]
$$

This is the central mathematical move in TBI: the system optimizes against a measured or learned proxy of deployment cost.

---

## 4. What Is and Is Not Being Optimized

### 4.1 Optimized Variables

TBI can optimize over:

- route selection
- expert selection
- early-exit thresholds
- checkpoint placement
- quantization policy
- compression level
- model family choice
- batching and scheduling policy, if included in the action space

### 4.2 Not Directly Optimized in the Literal Sense

TBI does **not** require or assume exact analytical gradients of:

- junction temperature with respect to arbitrary weights
- vendor-reported board power with respect to a single forward pass
- cache-controller or memory-scheduler behavior with respect to hidden states

These quantities can still influence optimization through traces, calibration, and surrogate learning.

---

## 5. Time-Scale Separation

One reason the literal thermodynamic picture is misleading is that the relevant variables operate on different time scales.

- model outputs and routing decisions occur per request or per token
- queueing and batching evolve over milliseconds
- board power is often sensor-averaged over milliseconds or longer
- thermal headroom changes more slowly and has path dependence
- infrastructure constraints can change over minutes to hours

TBI therefore benefits from a **time-scale separation**:

1. fast loop: routing and serving decisions
2. medium loop: threshold tuning and scheduler adjustment
3. slow loop: offline retraining, compression, and policy updates

This separation is both scientifically cleaner and operationally safer.

---

## 6. Risk-Constrained Interpretation

A useful interpretation of TBI is not “minimize temperature” but rather:

- minimize expected task loss
- minimize energy and latency at fixed quality
- keep thermal and throttling risk below a target level

This can be expressed as:

$$
\min \;
\mathbb{E}[\mathcal{L}_{\text{task}}]
\quad \text{s.t.} \quad
\mathbb{E}[J] \leq B_J,
\;
\text{p95}(L) \leq B_{L,95},
\;
\Pr(\chi = 1) \leq \epsilon
$$

This risk-constrained formulation is more appropriate than treating all hardware variables as directly trainable signals.

---

## 7. Offline Update Loop

The TBI optimization loop proceeds as follows.

1. collect traces from deployment
2. estimate route- and workload-specific costs
3. fit or update surrogate models
4. generate candidate routing or compression policies
5. validate candidates on held-out tasks and held-out traces
6. promote only if they improve the operator’s objective

This makes TBI a closed-loop program without requiring unsafe live mutation.

---

## 8. Failure Modes and Limitations

### 8.1 Distribution Shift

Surrogates trained on one workload may misestimate cost on another. Cost-model validation on held-out traces is therefore mandatory.

### 8.2 Sensor Noise and Aggregation

Power and temperature sensors often report averaged values, not true per-request measurements. Overconfident attribution will produce misleading gradients or rankings.

### 8.3 Coupling to Queueing and Batching

The cost of a route is not purely intrinsic to the route; it depends on batch composition, concurrency, and other traffic. TBI must evaluate routes in realistic serving conditions.

### 8.4 Conservative Use of Thermal Variables

Thermal headroom is highly useful for control and safety, but weak as a precise credit-assignment target. For most deployments, energy, latency, and memory pressure will be better primary optimization targets.

---

## 9. Testable Hypotheses

A publication-grade TBI program should make falsifiable claims.

1. **Routing hypothesis:** A calibrated multi-path policy can reduce expected serving cost at fixed quality on skewed workloads.
2. **Compression hypothesis:** Trace-guided distillation or pruning can outperform static compression chosen without deployment traces.
3. **Control hypothesis:** Telemetry-aware thresholds can reduce throttling and tail latency during load spikes.
4. **Hardware hypothesis:** Custom accelerators with finer-grained parameter residency control can shift the achievable quality-cost frontier further than software-only routing.

---

## 10. Summary

The mathematically strongest version of TBI is not exact thermodynamic differentiation. It is a framework for optimizing task quality under physical budgets using adaptive inference, telemetry-derived cost estimation, and closed-loop offline updates. That framing is both rigorous enough for scientific evaluation and broad enough to guide systems design.
