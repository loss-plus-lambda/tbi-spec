# TBI Whitepaper Appendix E: Background Optimization and Maintenance Pipeline

> **Document:** `docs/05_background_daemons.md`  
> **Depends on:** [docs/01_mathematical_foundations.md](01_mathematical_foundations.md), [docs/04_tier3_sparse_workers.md](04_tier3_sparse_workers.md)  
> **Related:** [docs/02_tier1_telemetry.md](02_tier1_telemetry.md), [docs/03_tier2_routing.md](03_tier2_routing.md)

---

## Overview

This appendix revises the original “background daemons” concept into a safer and more scientifically credible **background optimization and maintenance pipeline**. The key change is operational: production systems should not be described as freely self-editing organisms that mutate their active weights in place. In practice, model changes that affect correctness belong in an offline or shadow pipeline with evaluation, canaries, rollback, and version control.

This does not weaken the TBI idea. It strengthens it. The core insight remains that deployment traces can guide ongoing optimization. The difference is that the optimization loop is treated as a disciplined engineering pipeline rather than as unconstrained autonomous self-modification.

---

## 1. Separation of Online and Offline Actions

A production TBI system should distinguish three classes of actions.

### 1.1 Safe Online Maintenance

These actions can happen within the serving system with minimal risk:

- cache eviction and compaction
- retrieval-index maintenance
- route-statistics aggregation
- telemetry summarization
- threshold updates within prevalidated bounds

### 1.2 Shadow or Offline Optimization

These actions can change model behavior and therefore belong outside the active serving path:

- pruning
- quantization policy changes
- distillation
- architecture search over routing or exits
- expert-specialization updates
- route recalibration beyond prevalidated ranges

### 1.3 Promotion and Rollback

Candidates that perform well offline should be:

1. deployed in shadow mode
2. evaluated against live traffic without serving users
3. promoted through canary rollout
4. rolled back immediately if regression thresholds are exceeded

This is the proper operational interpretation of TBI’s self-improvement loop.

---

## 2. Trace-Guided Optimization Loop

A disciplined TBI update cycle looks like this.

### Step 1: Collect Traces

Collect route-level records containing:

- request features
- selected path
- latency
- energy proxies
- memory-pressure metrics
- throttle indicators
- quality labels or reference outputs, when available

### Step 2: Identify Bottlenecks

Aggregate traces to find:

- routes with poor quality-per-cost
- workload slices that systematically trigger the expensive path
- experts or checkpoints that underperform
- candidates for compression or recalibration

### Step 3: Generate Candidates

Candidate changes may include:

- a distilled small model
- a pruned or quantized heavy path
- different early-exit thresholds
- alternative route policies
- improved domain specialization
- better cache or retrieval policies

### Step 4: Evaluate Offline

Evaluate candidates on:

- held-out task data
- held-out deployment traces
- route-specific stress tests
- budget-constrained benchmarks

### Step 5: Shadow and Canary

Only candidates that survive offline evaluation should be run in shadow or canary mode.

---

## 3. Compression Modules

### 3.1 Pruning and Quantization

Pruning and quantization remain core TBI tools, but the whitepaper should frame them conservatively.

Strong claim:

> Deployment traces can help prioritize which components are worth compressing.

Weak claim that should be avoided:

> The system can safely prune itself online with no external controls.

Pruning candidates should therefore be scored using both:

- task-salience signals
- deployment-cost signals

### 3.2 Distillation

Distillation is one of the strongest tools in the TBI framework. A smaller student can be trained to mimic the heavy path on the workload slices that actually matter in deployment.

Useful TBI question:

- can trace-targeted distillation outperform generic compression chosen without deployment data?

That is a clear and publishable hypothesis.

---

## 4. Index and Memory Maintenance

The original TBI concept includes consolidation of repeated experience into more efficient retrieval or cache structures. In modern systems language, this is best described as:

- retrieval-index maintenance
- cluster or centroid updates
- cache compaction
- route-aware memory management

This is a valuable part of TBI and one of the safest forms of continuous optimization, because it need not alter model weights directly.

---

## 5. Promotion Criteria and Scheduling Constraints

A candidate should be promoted only if it satisfies all explicit gates simultaneously.

**Quality, latency, and safety gates:**

$$
\Delta Q \leq \epsilon_Q,
\qquad
\Delta J \leq -\epsilon_J,
\qquad
\Delta L_{95} \leq \epsilon_L,
\qquad
\Delta \chi \leq 0
$$

where:

- $\Delta Q$ is quality regression relative to baseline
- $\Delta J$ is change in serving energy cost
- $\Delta L_{95}$ is change in p95 latency
- $\Delta \chi$ is change in safety or throttling violations

**Net-negative Joule amortization gate.** A candidate that passes the above gates but whose training cost exceeds its projected serving savings is a net energy consumer and must not be promoted:

$$\frac{E_{\text{daemon}}}{|\Delta J_{\text{serving}}| \cdot R_{\text{projected}}} \leq H_{\text{amortize}}$$

where $E_{\text{daemon}}$ is the measured energy cost of the daemon run, $|\Delta J_{\text{serving}}|$ is the measured per-request energy improvement, $R_{\text{projected}}$ is the expected request count over the model's remaining deployment window, and $H_{\text{amortize}}$ is a configurable maximum amortization horizon (default: 30 days). All five gates must pass before promotion proceeds.

**Diminishing-returns stopping criterion.** Iterative daemon passes exhibit concave marginal returns: successive compression or pruning rounds recover less redundancy at constant compute cost. Daemon cycles terminate automatically when marginal per-request energy improvement falls below an operator-defined threshold $\kappa$:

$$\frac{\partial\,|\Delta J_{\text{saving}}|}{\partial\,n_{\text{rounds}}} \leq \kappa$$

Running daemon iterations past this boundary inverts the energy ROI and is treated as a configuration defect.

**Grid-Aware Efficiency Policy.** Energy accounting must not conflate Joule consumption with carbon impact. Off-peak scheduling that minimizes GPU idle time can worsen grid carbon intensity when those hours are served by baseload carbon generation. TBI therefore mandates grid-aware scheduling for daemon windows:

- Daemon execution is preferred when marginal grid carbon intensity $\gamma(t)$ is at or below the rolling 24-hour median $\bar{\gamma}_{24h}$, sourced from a real-time forecast (e.g., WattTime or Electricity Maps).
- The effective scheduling metric is the carbon-adjusted cost $E_{\text{carbon}} = E_{\text{daemon}} \cdot \gamma(t)$, not raw Joules alone.

**Thermal Recovery Gap.** Daemon runs at full GPU utilization delay thermal recovery of the package. A mandatory cooldown gap $\Delta t_{\text{cool}}$ is enforced between the end of any daemon window and the start of the subsequent peak traffic window:

$$\Delta t_{\text{cool}} \geq R_\theta C_\theta \ln\!\left(\frac{\tau_{\text{daemon}} - \tau_{\text{ambient}}}{\tau_{\text{target}} - \tau_{\text{ambient}}}\right)$$

where $R_\theta$ is the package thermal resistance, $C_\theta$ is thermal capacitance, $\tau_{\text{daemon}}$ is the measured junction temperature at daemon completion, and $\tau_{\text{target}}$ is the peak-traffic thermal headroom target. Where hardware thermal models are unavailable, this window defaults to empirical measurement across at least 20 run-cooldown cycles on the target hardware class.

This makes the update policy measurable, auditable, globally energy-justified, and carbon-aware.

---

## 6. Governance and Reproducibility

A publication-grade TBI system should include:

- model change records
- route-policy change records
- reproducible offline evaluation
- canary analysis
- rollback procedures
- trace retention and privacy policy
- clear ownership over promotion decisions

Without these elements, the “self-optimization” story becomes operationally fragile.

---

## 7. What Remains Autonomous

Even with this conservative rewrite, TBI still supports meaningful autonomy.

The system can autonomously:

- collect telemetry
- summarize route statistics
- generate optimization jobs
- rank candidate improvements
- run offline benchmarks
- recommend promotions

What it should not autonomously do in the strongest sense is alter active production weights without evaluation and deployment safeguards.

---

## 8. Research Questions

A strong TBI paper should pose concrete questions for the background pipeline.

1. Does trace-guided compression outperform static compression?
2. Which workload slices benefit most from route-specific distillation?
3. How often should route policies be recalibrated under nonstationary traffic?
4. When does retrieval or cache maintenance produce larger gains than model compression?
5. How much of the TBI benefit comes from routing versus offline optimization?

These are precise, falsifiable questions.

---

## 9. Summary

The background pipeline is where TBI becomes a closed-loop engineering discipline. Its strongest form is not “the model edits itself while serving users.” Its strongest form is a continuous, trace-driven optimization process with explicit evaluation gates and safe rollout. That version is both publishable and operationally credible.