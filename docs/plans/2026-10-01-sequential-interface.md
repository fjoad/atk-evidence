# Resolve and test the decoder-to-GRU interface

Date: 2026-10-01. State: source review in progress.

The user approved the proposed source review, a separately documented minimal
interface interpretation/control, its fixed learning checks and continuation
toward the bounded GPU timing/pilot only if validation passes. Preserve all
earlier ReLU results and do not silently relabel their implementation.

1. Recheck complete relevant target pages (Fig.2, TableII, IV-B/C, Algorithm1)
   against the earlier complete-paper reading; inspect a cited source only
   where it actually informs the missing interface. Record what is explicit,
   what remains ambiguous and the finite next completion before any output.
2. Implement the admitted interface choice(s) explicitly in the direct model,
   retaining the old ReLU default/serialized behavior. Check shapes, weights,
   feedback, classification gradients and fresh persistence.
3. Freeze and run bounded full-width constructed learning comparisons, keeping
   data, seed, objective, optimizer, schedule and evaluation fixed. Preserve
   every result; do not add branches after seeing which performs best.
4. If a predeclared source interpretation passes the gate, proceed with the
   already approved20-minute constructed GPU timing stage and then the180-minute
   fixed p00/p30 research pair if timing/audits pass. Actual research work stays
   on cluster compute nodes. No silent guard extension or local fallback.
5. Audit, record exact findings/limits and current next decision, verify and
   commit. No website edit or publication; other studies remain untouched.

The scientific model definition and branch list below will be completed
before implementation or learning results, after the source check.

## Source decision and frozen comparison

Complete target pages2678/2680/2682/2683 were visually rechecked, with the
original PDF hash unchanged. The readout shape/equation remains omitted.
The source does not uniquely specify the interface. See
[SEQUENTIAL_INTERFACE.md](../../studies/takiddin-2021-robust-poisoning/SEQUENTIAL_INTERFACE.md).

Declare I-SEQ-scalar-sigmoid as the next numerical candidate from TableII's
component-output setting, and C-SEQ-scalar-linear as a separate clipping
control. Keep all other settings/data/seed unchanged; retain the ReLU default
and original artifacts. Each activation affects feedback as well as GRU
input. Four fixed full-width,8-step,300-update learning fits,300s guard each,
no additional branch or seed after results. Same >=90%/BCE<log(2)/2 criterion.
Only a passing Sigmoid pair can promote to the previously approved GPU
timing/pilot sequence. A better linear-control metric does not select it.
