# ORCA Incident Error Analysis Report

## 1. Executive Diagnostic Summary

- **Total Windows Evaluated:** 288
- **False Positive Count:** 4 (FPR: 0.0278)
- **False Negative Count:** 6 (FNR: 0.0417)
- **Root-Cause Mislocalizations:** 3

---

## 2. False Positive Characterization
False positives typically occur during workload ramp-up periods where transient metric spikes (e.g. CPU saturation during batch request processing) resemble injected faults, but logs and traces show normal HTTP 200 responses.

## 3. False Negative Characterization
False negatives concentrate immediately following fault injection ($t = T_{inject}$), where distributed cascading effects take 15–30 seconds to propagate across downstream service dependencies before manifesting clearly in aggregate metric distributions.

## 4. Root-Cause Error Diagnostics
When root-cause localization predicts a neighbouring service, it is primarily due to cascading latency propagation (e.g., an upstream caller service `checkoutservice` recording high trace latency when its callee `paymentservice` is experiencing CPU starvation).
