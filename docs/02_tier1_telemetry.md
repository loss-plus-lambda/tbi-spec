# TBI Whitepaper Appendix B: Telemetry, Measurement, and Control

> **Document:** `docs/02_tier1_telemetry.md`  
> **Depends on:** [docs/01_mathematical_foundations.md](01_mathematical_foundations.md)  
> **Related:** [docs/03_tier2_routing.md](03_tier2_routing.md), [docs/05_background_daemons.md](05_background_daemons.md)

---

## Overview

This appendix defines the telemetry layer for TBI. Its purpose is not merely to collect numbers, but to support a measurement model for deployment-aware optimization. Telemetry is treated according to its actual limitations.

On commodity accelerators, telemetry is often coarse, averaged, vendor-specific, and partially asynchronous with request execution. As a result, not every deployment metric is equally suitable for fine-grained optimization. TBI therefore prioritizes **energy, latency, throughput, memory pressure, and throttling events** as primary serving metrics, while treating **temperature and clock state** as slower control and safety signals.

---

## 1. Measurement Goals

The telemetry layer should answer five operational questions:

1. How much energy does the system consume per request or per token?
2. Which routes or model variants dominate latency and tail latency?
3. Where does memory bandwidth or cache pressure become the bottleneck?
4. When and why does the hardware throttle?
5. Which signals are reliable enough to drive routing or offline optimization?

These questions are more useful than attempting exact per-request thermodynamic attribution in environments where sensors do not support it.

---

## 2. Metric Hierarchy

### 2.1 Primary Optimization Metrics

The preferred optimization targets are:

- **Joules per request or token**
- **mean and tail latency**
- **throughput under realistic load**
- **memory bandwidth utilization or bytes moved**
- **throttle-event rate**

These are the metrics most directly tied to service quality and deployment cost.

### 2.2 Secondary State and Safety Metrics

The following remain important, but should usually be treated as state or constraint variables:

- junction temperature
- clock frequency
- power limit state
- fan state
- thermal headroom

These metrics are valuable for control and diagnosis, but often too slow or noisy for precise example-level optimization.

---

## 3. Time-Scale Separation and Dual-Window Architecture

Mixing incompatible signal time scales is the primary source of attribution error in hardware-aware ML. TBI resolves this by enforcing a **dual-window telemetry architecture** with two strictly separated polling loops.

### 3.1 Fast Window ($\leq 1$ ms)

Gathered at kernel or micro-batch granularity with zero sensor lag. These signals form the **primary optimization substrate**:

- kernel duration
- FLOP counts (from hardware performance counters or model metadata)
- route-level request timestamps
- achieved memory bandwidth

### 3.2 Slow Window ($\geq 5\text{–}50$ ms)

Many vendor telemetry APIs expose values at millisecond or multi-millisecond resolution. These signals are consumed **exclusively as control and safety inputs** — they must not be used as per-request cost targets:

- board power
- average clocks
- memory utilization snapshots
- junction temperature
- thermal headroom
- throttling indicators

### 3.3 Architectural Requirement

Conflating the two windows — for example, attributing a vendor-reported 10 ms power average to a 3 ms token generation event — produces systematic attribution errors that corrupt both surrogate cost models and routing policies. The two-window separation is a **hard architectural requirement**, not a recommendation. Any implementation that polls slow-window signals on the fast-window path is non-conformant.

---

## 4. Practical Measurement Tiers

### 4.1 Tier A: Portable Commodity Measurement

This is the default for production systems on mainstream GPUs.

Available signals often include:

- average board power
- temperature
- clock frequency
- coarse memory utilization
- throttling indicators

These are sufficient for route ranking, threshold adaptation, and offline cost modeling, but not for exact request-level causal attribution.

### 4.2 Tier B: Profiler-Assisted Sampling

Periodic profiling runs can add:

- memory transaction counters
- occupancy or stall indicators
- kernel-level timing
- achieved bandwidth

These are useful for calibrating surrogates and understanding route bottlenecks.

### 4.3 Tier C: Hardware Research Instrumentation

Custom accelerators or privileged kernel modules may expose finer-grained counters. Only at this tier does sub-millisecond control become a central design assumption.

The whitepaper therefore distinguishes:

- **portable claims** for commodity hardware
- **research claims** for specialized hardware

---

## 5. Recommended Telemetry Vector

A practical telemetry state vector is:

$$
\mathbf{s}(t) =
\left[
P(t),\;
\omega(t),\;
u_M(t),\;
\chi(t),\;
T(t),\;
q(t)
\right]
$$

where:

- $P(t)$ is power or an energy proxy
- $\omega(t)$ is clock state
- $u_M(t)$ is memory or bandwidth utilization
- $\chi(t)$ is a throttling indicator or safety flag
- $T(t)$ is temperature or thermal headroom
- $q(t)$ is queueing state such as backlog or batch size

Including queue state is important because serving cost depends on traffic and batching, not only on model internals.

---

## 6. Attribution Strategy

### 6.1 Exact Attribution Is Rarely Available

On most production hardware, it is not scientifically sound to claim exact causal attribution of temperature or board power to a single request.

### 6.2 Useful Alternatives

Instead, TBI should use:

- route-level averages over windows
- A/B comparisons between route policies
- micro-benchmarks for fixed prompts and batch shapes
- periodic profiling runs for representative slices
- counterfactual replay on held-out traces

### 6.3 Energy Estimation: Analytical Proxy (Primary)

The preferred primary energy estimator is the **FLOP-count analytical proxy**, which is deterministic, zero-latency, and immune to sensor jitter or idle-power baseline drift:

$$\hat{J}_k = \alpha_k \cdot \text{FLOP}(x,\, k) + \beta_k$$

where $\alpha_k$ is a per-route joules-per-FLOP coefficient and $\beta_k$ is a per-route memory-access cost term, both calibrated from periodic offline profiling runs at representative batch shapes. This proxy is the default estimator for routing and surrogate training.

### 6.4 Windowed Power Integration (Fallback)

When a FLOP proxy cannot be derived — for example, on closed-source model paths or novel architectures — power integration over windows provides a lower-fidelity fallback:

$$
J_{\text{window}} \approx \int_{t_0}^{t_1} \left(P(t) - P_{\text{idle}}\right) dt
$$

Three limitations must be acknowledged when using this fallback: (1) $P_{\text{idle}}$ is not a stable constant at ms resolution; (2) the integration window must align with the slow-window cadence, not the fast-window cadence; (3) allocation from windows to routes must be done statistically over many samples, never presented as request-level ground truth.

---

## 7. Telemetry-Guided Control

TBI uses telemetry in two main ways.

### 7.1 Fast Online Control

The serving stack may adjust:

- routing thresholds
- maximum depth or early-exit thresholds
- batch-size caps
- concurrency settings
- model-choice policy under pressure

These adjustments should be conservative and reversible.

**Sensor-Delay Compensation.** The routing gate consumes $\vec{T}(t - \Delta t_{\text{sensor}})$, a delayed observation with $\Delta t_{\text{sensor}} \in [1, 10]$ ms on commodity accelerators. Threshold updates must be passed through a low-pass filter with cutoff frequency:

$$\omega_c \leq \frac{1}{2\,\Delta t_{\text{sensor}}}$$

The DVFS controller gain $K$ has no closed-form bound without hardware-specific plant identification. A conservative starting estimate for a linearized first-order thermal plant with pure sensor delay is:

$$K \leq \frac{1}{4\,\omega_c\,\Delta t_{\text{sensor}}\,R_\theta}$$

where $R_\theta$ is the junction-to-case thermal resistance (°C/W) at the nominal operating clock state. This estimate provides approximately 45° phase margin and must be empirically validated before production deployment: apply a 10°C step in $\tau_{\text{junction}}$ at nominal load and confirm settling within $5\,R_\theta C_\theta$ with less than 15% overshoot. Any gain selection that fails this step-response test must not be deployed.

### 7.2 Slow Offline Optimization

Trace data supports:

- surrogate cost-model fitting
- route recalibration
- compression and distillation targeting
- identification of workload slices that are expensive or underperforming

---

## 8. Measurement Claims: Prohibited and Specified

### 8.1 Claims That Remain Prohibited

The following claims must not be made without hardware-specific evidence:

- a universal sub-500 microsecond telemetry loop on commodity accelerators
- exact per-request thermal attribution using vendor APIs alone
- the claim that vendor-reported power draw is synchronized tightly enough for token-level ground truth
- the claim that the telemetry stack is guaranteed non-interfering at all cadences

### 8.2 Claims That Are Now Formally Specified

The following are defined as hard architectural requirements and may be positively asserted when the implementation conforms:

- the dual-window separation between fast ($\leq 1$ ms) and slow ($\geq 5\text{–}50$ ms) telemetry paths
- the FLOP-count analytical proxy $\hat{J}_k = \alpha_k \cdot \text{FLOP}(x,k) + \beta_k$ as the primary per-route energy estimator
- the low-pass filter bound $\omega_c \leq 1 / (2\,\Delta t_{\text{sensor}})$ on DVFS controller updates
- the empirically validated DVFS controller gain bound (45° phase-margin step-response test, §7.1), with conservative starting estimate $K \leq 1/(4\,\omega_c\,\Delta t_{\text{sensor}}\,R_\theta)$

A conformant TBI telemetry implementation satisfies all four specified requirements. Measurement resolution and overhead beyond these bounds remain hardware-tier dependent.

---

## 9. Evaluation Metrics

The telemetry layer should be judged by whether it supports better deployment decisions. Relevant metrics include:

- route ranking stability
- predictive accuracy of cost surrogates
- reduction in throttle events after control adaptation
- improvement in joules per token or request
- improvement in p95 latency under the same quality target

If telemetry does not improve these outcomes, it is instrumentation overhead without systems value.

---

## 10. Summary

The TBI telemetry layer is best understood as a measurement-and-control substrate, not as a magical exact readout of thermodynamic gradients. Its strongest form on current hardware is a practical stack for collecting route-level cost statistics, informing conservative online control, and generating better offline optimization targets.
