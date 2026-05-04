---
name: Feature request
about: Suggest a new capability for LAPPATO_MCB
title: "[feature] "
labels: enhancement
assignees: ''
---

## Use case

<!-- Describe the workflow you want to support and why current
LAPPATO_MCB cannot. Include the manifest / domain / pipeline shape. -->

## Proposed change

<!-- Sketch the API or behaviour you have in mind. Be concrete:
function signature, manifest field, CLI flag, etc. -->

## Compatibility with the design constraints

LAPPATO_MCB is opinionated. Please confirm the proposal respects each
of the following (see `CONTRIBUTING.md` for details):

- [ ] Stays within the **stdlib-only** core (no new mandatory deps).
- [ ] Preserves **determinism** (same input + state → same output).
- [ ] No paid services or hosted backends required by default.
- [ ] Keeps the **two-line integration** (`start()` / `stop()`).
- [ ] Activation/threshold logic remains **auditable in code**.

If a constraint cannot be satisfied, please explain the trade-off
explicitly so reviewers can weigh it.

## Alternatives considered

<!-- What else did you consider, and why is the proposal preferable? -->

## References

<!-- Optional: papers, RFCs, prior art that justify the proposal. -->
