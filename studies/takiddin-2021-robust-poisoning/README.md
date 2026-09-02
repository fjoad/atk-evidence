# Study 3: robust electricity-theft detection under label poisoning

This study audits:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, “Robust Electricity
> Theft Detection Against Data Poisoning Attacks in Smart Grids,” *IEEE
> Transactions on Smart Grid*, 12(3), 2675–2684, 2021.

DOI: `10.1109/TSG.2020.3047864`

## Current state

The source-only audit is complete and stopped for discussion. The paper was
identified, fingerprinted, and visually inspected in full; the method, causal
claims, and Tables II–V are frozen. A preregistered audit found that three
sequential-ensemble rows cannot reconcile DR, FA, PR, and ACC at any class
prevalence within one-decimal rounding. This is a source-level internal
inconsistency, not a trained non-reproduction or an inference about intent. No
dataset has been prepared and no model has been implemented, trained, or
scored for this study.

The study is independent of the earlier electricity-theft audit. Shared
authors, data, attacks, or terminology may motivate checks but cannot transfer
a result or verdict.

## Read next

- [`METHOD.md`](METHOD.md): paper-derived experiment and causal specification.
- [`SOURCE_AUDIT_CONTRACT.md`](SOURCE_AUDIT_CONTRACT.md): checks frozen before
  the printed values are analyzed.
- [`SOURCE_AUDIT_FINDING.md`](SOURCE_AUDIT_FINDING.md): source-only arithmetic
  result, limitations, and compute boundary.
- [`EXPLANATION_REGISTER.md`](EXPLANATION_REGISTER.md): live competing
  explanations and discriminating evidence.
- [`../../docs/plans/2026-09-02-robust-poisoning-source-audit.md`](../../docs/plans/2026-09-02-robust-poisoning-source-audit.md): active authorization and stop rule.

## Experimental boundary

The paper states 50 epochs, batch size 100, an NVIDIA GeForce RTX 2070, and
approximate training times of one hour for shallow detectors, 1.5–3 hours for
deep detectors, three hours for ensemble averaging, and four hours for the
sequential ensemble. A later numerical attempt must treat those statements as
part of the target rather than escaping a mismatch with newer or additional
hardware.

The source-only stop is now active. Data preparation and all training require a
new reviewed contract and cluster authorization.
