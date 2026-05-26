# TBI Whitepaper Appendix B: Telemetry, Measurement, and Control

> **Document:** `docs/02_tier1_telemetry.md`  
> **Depends on:** [docs/01_mathematical_foundations.md](01_mathematical_foundations.md)  
> **Related:** [docs/03_tier2_routing.md](03_tier2_routing.md), [docs/05_background_daemons.md](05_background_daemons.md)

---

## Overview

This appendix defines the telemetry layer for TBI. Its purpose is not merely to collect numbers, but to support a defensible measurement model for deployment-aware optimization. The most important scientific adjustment in this whitepaper is that telemetry is treated according to its actual limitations.

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

## 3. Time-Scale Separation

A major source of confusion in hardware-aware ML is the mixing of incompatible time scales.

### 3.1 Fast Signals

Some counters can be gathered at kernel or micro-batch granularity:

- kernel duration
- FLOP estimates
- profiler counters
- route-level request timestamps

### 3.2 Medium Signals

Many vendor telemetry APIs expose values at millisecond or multi-millisecond resolution:

- board power
- average clocks
- memory utilization snapshots

### 3.3 Slow Signals

Thermal state usually evolves more slowly and depends on recent history:

- junction temperature
- thermal headroom
- throttling due to sustained heating

TBI should exploit this separation rather than pretending it does not exist.

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

### 6.3 Windowed Energy Estimation

When power is available only as a time series, request energy can be approximated by integration over windows:

$$
J_{\text{window}} \approx \int_{t_0}^{t_1} \left(P(t) - P_{\text{idle}}\right) dt
$$

Allocation from windows to routes should then be done statistically, not presented as exact ground truth.

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

### 7.2 Slow Offline Optimization

Trace data supports:

- surrogate cost-model fitting
- route recalibration
- compression and distillation targeting
- identification of workload slices that are expensive or underperforming

---

## 8. Measurement Claims That Should Be Avoided

To keep the whitepaper scientifically tight, the following claims should be avoided unless backed by hardware-specific evidence:

- a universal sub-500 microsecond telemetry loop on commodity accelerators
- exact per-request thermal attribution using vendor APIs alone
- the claim that vendor-reported power draw is synchronized tightly enough for token-level ground truth
- the claim that the telemetry stack is guaranteed non-interfering at all cadences

A more defensible statement is that measurement resolution and overhead depend on the hardware and instrumentation level.

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
