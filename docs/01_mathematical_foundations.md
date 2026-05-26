# TBI Whitepaper Appendix A: Mathematical Foundations

> **Document:** `docs/01_mathematical_foundations.md`  
> **Depends on:** [README.md](../README.md)  
> **Related:** [docs/02_tier1_telemetry.md](02_tier1_telemetry.md), [docs/03_tier2_routing.md](03_tier2_routing.md)

---

## Overview

This appendix formalizes Thermodynamically Bounded Intelligence as a problem of **constrained adaptive inference** and **surrogate-guided optimization**. The goal is not to claim that thermal sensors, board power, or memory controller behavior are exactly differentiable with respect to arbitrary model weights during standard training. The goal is narrower: deployment cost can be modeled, predicted, and incorporated into optimization over the controllable decisions of a serving system.

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

### 2.3 Lagrangian Relaxation and Discrete Routing

Route selection, early-exit depth, expert assignment, and quantization level are inherently discrete decisions. The standard Lagrangian relaxation of a mixed-integer program does not generally yield integer-feasible solutions; the duality gap can be nonzero and problem-dependent. TBI therefore adopts a **stochastic policy formulation** for the routing layer:

$$k \sim \pi_\phi(x, s) = \text{Gumbel-Softmax}\!\left(g_\phi(x, s),\; \tau_{\text{temp}}\right)$$

where $g_\phi(x, s)$ are the gate logits and $\tau_{\text{temp}}$ is a temperature parameter annealed during offline policy optimization. The Gumbel-Softmax reparameterization provides differentiable surrogate gradients through the discrete routing decision during training. At deployment, the policy is evaluated in hard mode (argmax). For multi-step escalation trees where Gumbel-Softmax is structurally inapplicable, the REINFORCE estimator with a learned variance-reducing baseline is used instead.

The Lagrangian relaxation is then applied to the **continuous** policy parameters and surrogate cost terms:

$$
\min_{\theta, \phi, a}
\;\mathbb{E}
\left[\mathcal{L}_{\text{task}} + \lambda_J J + \lambda_L L + \lambda_M M + \lambda_\chi \chi\right]
$$

This relaxation makes explicit that TBI is a **multi-objective optimization problem** over quality and deployment cost. The Lagrange multipliers $\{\lambda_k\}$ are adapted by a derivative-damped dual ascent update with hysteresis to prevent oscillation under delayed cost feedback (see §7).

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

trained by minimizing a variance-normalized composite loss:

$$
\min_{\psi}
\sum_{i=1}^{N}
\left\|
\mathbf{D}^{-1/2}\!\left(\hat{\mathbf{c}}_\psi(z_i) - c_i\right)
\right\|_2^2 + \eta \|\psi\|_2^2
$$

where

$$\mathbf{D} = \mathrm{diag}(\hat{\sigma}_J^2, \hat{\sigma}_L^2, \hat{\sigma}_M^2, \hat{\sigma}_\chi^2)$$

is the diagonal matrix of empirical per-component variances measured on $\mathcal{D}_{\text{trace}}$. Without this normalization the cost component with the largest absolute scale (typically $M$, memory bandwidth in bytes, $\mathcal{O}(10^9)$, versus $J$ in millijoules, $\mathcal{O}(10^{-3})$) dominates the MSE objective and the surrogate fails to learn the other components.

The binary throttling indicator $\chi \in \{0,1\}$ is trained with a separate **cross-entropy head**:

$$
\min_{\psi_\chi}
\sum_i
\left[-\chi_i \log \hat{\chi}_{\psi_\chi}(z_i) - (1-\chi_i)\log(1 - \hat{\chi}_{\psi_\chi}(z_i))\right] + \eta_\chi \|\psi_\chi\|_2^2
$$

The surrogate uses separate prediction heads: normalized MSE for continuous costs $(J, L, M)$ and cross-entropy for the binary safety indicator $\chi$.

### 3.4 Use in Optimization

The deployment-aware objective then becomes:

$$
\min_{\theta, \phi, a}
\;\mathbb{E}
\left[\mathcal{L}_{\text{task}} + \lambda^\top \hat{\mathbf{c}}_\psi(\Phi(x, \pi_\phi(x, s), s, a))\right]
$$

This is the central mathematical move in TBI: the system optimizes against a measured or learned proxy of deployment cost.

### 3.5 Surrogate Validity Under Policy Updates

Surrogate retraining uses the following notation:

- $\hat{\mathbf{c}}_\psi$: surrogate cost model
- $\pi_\phi^{\text{old}}$: prior routing policy
- $\pi_\phi^{\text{new}}$: updated routing policy

Traces are collected under $\pi_\phi^{\text{old}}$. After updating to $\pi_\phi^{\text{new}}$, the distribution of feature vectors $z_i$ shifts and the surrogate may become miscalibrated on the new request distribution.

**Retraining Trigger.** Surrogate retraining is triggered when routing distribution shift exceeds a calibrated KL threshold:

$$\mathbb{E}_{z \sim \mathcal{D}_{\text{trace}}}\!\left[\mathrm{KL}\!\left(\pi_\phi^{\text{new}}(\cdot \mid z) \;\Big\|\; \pi_\phi^{\text{old}}(\cdot \mid z)\right)\right] > \delta_{\mathrm{KL}}$$

An L2 norm on policy parameters is insufficient: it is reparameterization-sensitive. A large parameter change in an overparameterized region may shift the routing distribution negligibly, while a small change near a decision boundary may flip routing for a substantial fraction of requests. The KL trigger measures actual distribution shift rather than parameter-space distance.

**Calibrating $\delta_{\mathrm{KL}}$.** Set $\delta_{\mathrm{KL}}$ to the 95th percentile of this KL divergence measured over an ensemble of offline random-restart policy optimization experiments on a held-out workload slice. This places the threshold above typical optimization noise, so retraining fires only for policy updates that are unusually large relative to the optimization landscape.

**Retraining Procedure.** Use variance-normalized IW-ERM with clipped importance weights:

$$\min_\psi \sum_i w_i \left\|\mathbf{D}^{-1/2}\!\left(\hat{\mathbf{c}}_\psi(z_i) - c_i\right)\right\|_2^2 + \eta\|\psi\|_2^2, \qquad w_i = \mathrm{clip}\!\left(\frac{p_{\text{new}}(z_i)}{p_{\text{old}}(z_i)},\; \tfrac{1}{10},\; 10\right)$$

where $\mathbf{D}$ is the per-component variance matrix from §3.3, and the density ratio $p_{\text{new}}/p_{\text{old}}$ is estimated from a lightweight binary discriminator trained on old and new trace distributions. Clipping at $[0.1, 10]$ is mandatory: unclipped IS with a neural discriminator has unbounded variance when the old and new routing distributions have near-disjoint support, destabilizing the regression update.

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

TBI enforces a **three-tier control separation** aligned with the Dual-Window Telemetry Architecture (Appendix B):

1. **Fast serving loop** (sub-ms): routing decisions, gate evaluation, KV cache lookups — driven by fast-window telemetry ($\leq 1$ ms FLOP counters, timestamps, bandwidth)
2. **Slow control loop** ($5\text{–}50$ ms): DVFS threshold updates, batch-size cap tuning, safety flag propagation — driven by slow-window telemetry ($\geq 5\text{–}50$ ms thermal, board power, clock state)
3. **Offline optimization loop** (hours to weeks): surrogate retraining, Lagrange multiplier updates, policy gradient optimization, model compression and distillation — decoupled from real-time inference entirely; outputs enter production only through the evaluation and promotion pipeline (Appendix E, §7 here)

The fast and slow loops map directly to the two telemetry windows. The offline loop is not a real-time control loop — it is a batch optimization pipeline with weekly or longer cadence.

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

The TBI optimization loop runs once per deployment cycle (typically weekly) and is strictly decoupled from the real-time serving path.

1. **Collect traces** from the deployment period.
2. **Estimate costs**: aggregate route-level costs and constraint violations $e_k^{(t)} = \hat{C}_k^{(t)} - B_k$.
3. **Update or retrain surrogate** $\psi$: if the KL trigger fires (§3.5), run IW-ERM with clipped weights and variance normalization; otherwise run an incremental update on recent traces.
4. **Update Lagrange multipliers** $\{\lambda_k\}$ with PID-damped dual ascent and hysteresis.
5. **Optimize routing policy** $\phi$ by policy gradient through Gumbel-Softmax against the updated surrogate and multipliers, for $N_{\text{policy}}$ gradient steps, with $\theta$ and $\psi$ held fixed.
6. **Validate** candidate $\phi$ on held-out tasks and held-out deployment traces.
7. **Promote** $\phi$ only if it improves the operator objective and passes all evaluation gates (Appendix E §5).

For step 4, the update rule is:

$$
\lambda_k^{(t+1)} = \max\!\left(0,\; \lambda_k^{(t)} + \alpha_\lambda \cdot e_k^{(t)} + \mu \cdot \frac{e_k^{(t)} - e_k^{(t-1)}}{\Delta t} \right)
$$

where:

- $\alpha_\lambda \leq 1 / (L_\psi \cdot W)$
- $\mu \leq \alpha_\lambda \Delta t / 2$

Model weights $\theta$ are **never updated in this loop** — they belong exclusively to the compression and distillation pipeline (Appendix E). Jointly updating $\theta$ and $\phi$ here conflates routing adaptation with model mutation.

This makes TBI a closed-loop program without requiring unsafe live mutation.

---

## 8. Failure Modes and Limitations

### 8.1 Distribution Shift

Surrogates trained on one workload may misestimate cost on another. Cost-model validation on held-out traces is therefore mandatory.

### 8.2 Sensor Noise and Aggregation

Power and temperature sensors report averaged values at millisecond or multi-millisecond resolution, not true per-request measurements. The mitigation is twofold: (1) the **dual-window telemetry architecture** in Appendix B enforces a hard separation between fast signals (FLOP counters, $\leq 1$ ms) and slow signals (thermal, $\geq 5\text{–}50$ ms), preventing conflation across time scales; (2) the **FLOP-count analytical proxy** $\hat{J}_k = \alpha_k \cdot \text{FLOP}(x,k) + \beta_k$ replaces raw power integration as the primary energy estimator, eliminating the dependence on noisy sensor readouts for per-request cost attribution.

### 8.3 Coupling to Queueing and Batching

The cost of a route is not purely intrinsic to the route; it depends on batch composition, concurrency, and other traffic. TBI must evaluate routes in realistic serving conditions.

### 8.4 Conservative Use of Thermal Variables

Thermal headroom is highly useful for control and safety, but weak as a precise credit-assignment target. For most deployments, energy, latency, and memory pressure will be better primary optimization targets.

---

## 9. Testable Hypotheses

The TBI program defines falsifiable hypotheses:

1. **Routing hypothesis:** A calibrated multi-path policy can reduce expected serving cost at fixed quality on skewed workloads.
2. **Compression hypothesis:** Trace-guided distillation or pruning can outperform static compression chosen without deployment traces.
3. **Control hypothesis:** Telemetry-aware thresholds can reduce throttling and tail latency during load spikes.
4. **Hardware hypothesis:** Custom accelerators with finer-grained parameter residency control can shift the achievable quality-cost frontier further than software-only routing.

---

## 10. Summary

The mathematically strongest version of TBI is not exact thermodynamic differentiation. It is a framework for optimizing task quality under physical budgets using adaptive inference, telemetry-derived cost estimation, and closed-loop offline updates. That framing is both rigorous enough for scientific evaluation and broad enough to guide systems design.
