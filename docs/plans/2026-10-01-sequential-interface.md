# Resolve and test the decoder-to-GRU interface

Date: 2026-10-01. State: all bounded stages complete; pair407294 audited, stop before new compute.

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

## Constructed result and execution implementation

All four declared fits completed300 updates at100% held-out accuracy and
clipped BCE~1e-7. All share the old ReLU initial parameter hash; source/data/
weights/optimizer/metrics and fresh output/attention reload verify. Historical
ReLU defaults and outputs remain exact. Sigmoid therefore qualifies for
timing; linear remains a control. These are local software fixtures only.

The full48-step GPU preflight, direct50-epoch research runner, Table-V target
mapping and pair audit are implemented. They preserve original inputs and
historical functions. The new runner records initial/final representations,
raw class probabilities, all prescribed cutoff diagnostics, histories,
weights/configuration/norms and failures. It refuses an absent, failed,
wrong-source or altered preflight before loading research arrays.
No timings or research result have been produced yet. User reaffirmed local
small checks are acceptable and big jobs must use Panther. Cluster login is
restored, idle V100-16GB nodes were observed, existing neural environment and
prepared data are present, and the new preflight output path is clear.

## Cluster execution

Frozen scientific code3705bcca04279bbfa8523a573b9b573f123463b7 ran GPU
preflight407255: COMPLETED/0:0 in8:01 oncrimv3mgpu026. All14 cluster checks
passed. Epoch times58.0757/36.7454/36.9169s;135 updates and exact reload.
The runtime projection is1845.8433s per50 epochs, below3600s. Training
allocator peak1,570,985,216 bytes; this is not total process/driver VRAM.
Preflight SHA721256059d34a104fa2eda04ccb0bb8775603d01206117e4e82e1a9e54782c07
matches locally/remotely and the direct gate verifier passes.

The authorized real pair was submitted once as407294. Slurm held it pending
with PartitionTimeLimit: gpu-short permits only2h. Before fitting, the same
job was moved using scontrol to gpu-all, which supports the sameV10016GB
resources and the unchanged3h ceiling. The launcher partition is corrected
for future use; model/data/seed/epochs/70min fit guards are unchanged. This
is an operational scheduler correction, not a new fit or scientific branch.

## Completed outcome and next decision

Pair407294 completed0:0 in1:21:28 onPanther005. Both fixed fits completed
50epochs/2250updates in2125.66/2008.09s, with byte-identical initial weights.
Probabilities are constant0.504759/0.353959:DR/FA100/100 and0/0,AUC50/50.
No cutoff/reversal rescues the paper corners. P00 intermediate profile
variation does not reach the output; p30 representation is almostzero and
its lower MSE equals the zero baseline. Eight-step positive learning did not
establish useful48-step research learning. No causal or family-wide claim.

All15 copied files hashmatch, local pair/comparison audits match cluster
bytes, and source/input/metrics/history/serialized weights/config/optimizer/
norm and compute-node reload checks pass. Full runtime/results and two
read-only audit-operation corrections are preserved in the resultrecord.
No model inference ran locally; no website edits. Stop this pair. Next
propose bounded zero-fit activation/gradient inspection of the saved initial/
final states before any further training; it has not run. All previous
contracts, failed interfaces, preparations and artifacts remain preserved.
