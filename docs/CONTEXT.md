# ATK Evidence — Working Memory

**Last updated:** 2026-10-02

**Full48-step tanh learning gate verified:** scientific6ca174a; placement
recoveryfebc28e. Two tanh cases complete300updates each,100% held-out
accuracy/clippedBCE~1e-7; same initial weights and all other settings as the
ReLU reference. ReLU-normal becomes nonfinite at245, with a brief finite
loss dip to.489 at239; no plateau claim. ReLU-reversed is unrun under the
predeclared stop rule. Raw comparison stays failed/incomplete, not four-case
complete. Fresh-process GPU reload408787 passes both tanh models exactly;
partial artifact audit verifies failed reference without treating it as valid.
408764 pre-fit device error184s preserved;408778 stopped1088s;408787 audit32s;
total21:44 within35min. No CER inputs/fits. Alternative follows explicit tanh
formulas while retaining the IV-C ReLU conflict and other native-cell gaps.
Default ReLU/legacy behavior remains; only models.py gained the option.
Next specify/wire/timing-gate a fresh-weight tanh CER pair; current research
CLI still selects ReLU and old preflight does not authorize tanh. No automatic
launch/extra seed or website work. See [cell record](../studies/takiddin-2021-robust-poisoning/results/sequential_cells_20261002/README.md).

Raw data:sequential-cell-learning-20261002-attempt1 andattempt2. Attempt2
has two complete tanh fits, one preserved nonfinite ReLU fit, no reversedReLU.
All29 copiedfiles verify;15 oldpairfiles and failedattempt1 unchanged. First
corrected tanh initialization matches all failed-prefit arrays exactly. Fresh
GPU receiptSHAe93c1c80e0d39a169be71db51f8d3e7fd69ada4315971a3f299910831da77e3c.
GPU code:/export/home/fjoad/atk-evidence-seq-cells-retry-20261002 atfebc28e;
verification script frozen5d53338/copied bySHA. No active jobs. Model has
9,240,802params,32train/32test synthetic48steps, model/data seeds20260920/
20260926, Adam.001/BCErepair/no dropout/MaxNorm1/nativekernels unchanged.
20 pinned neuralchecks,405 repo passes/45skips, device regression and strict
semantic-equivalence data checks pass. PDF source unchanged/relevant pages
visually rechecked. No paper-data or mechanism conclusion follows from toy
learning. Never reuse synthetic trained weights for a proposed CER pilot.

**Earlier cell comparison recovery checkpoint:**408764 FAILED/1:0 in3:04, all cluster
fixtures passed but no full-width optimizer update ran. Broad GPU scope
incorrectly forced CPU-only TensorSliceDataset ontoGPU. Fix only placement:
GPU model variables, CPU input pipeline. Preserveattempt1 and require exact
initial-array parity before fittingattempt2. Same model/data/seed/settings;
retry31min + prior184s stays within original35min budget. No CER fit/data,
new branch, seed or website work. Recovery not yet submitted at this checkpoint.

**Full48-step cell comparison authorized and ready:** I-SEQ-tanh-cells
follows explicit Algorithm1 recurrent tanh formulas while preserving the
IV-C activation conflict. Only LSTM/GRU cell activation changes; Sigmoid
gates, scalar-Sigmoid bridge, ReLU Dense, all widths/optimizer/settings stay.
Default ReLU and legacy archives remain valid. Twenty pinned neural checks
and405 repository cases pass (45 environment skips). Four fixed synthetic
fits: tanh normal/reversed, then ReLU normal/reversed;32train/32test,48steps,
seed20260920,300updates each. One35min V10016GB/4CPU16GiB job,360s/fit.
No CER arrays/fits or automatic promotion. Freeze then submit once after
checking queue/output. No new job yet. See the recurrent-control plan.

Source recheck: complete pp2678/2679/2682/2683; source PDF hash unchanged.
New contract SEQUENTIAL_RECURRENT_CONTROL.md, plan2026-10-02-sequential-recurrent-control.md.
`cell_activation` is serialized relu/tanh, defaultrelu; not gate `recurrent_activation`.
Initial weights must match each other and priorGPU11c3bf3746800eebd9e695a9dda3abaafc9e7c445cbe8044a2110ca4da77f549.
Old four other direct files/functions remain unchanged; original artifacts
stay intact. New test initially mishandled nested symbolic output list; fixed
to m.outputs[0]. No numerical model change resolved that software-test error.
Largelearning onPanther; small localfixtures/read-only audits only. Website unchanged.

**Gate arithmetic complete:** ee1a420656cfa28d47008d12a550ea63f6145bd5,
Panther408551 COMPLETED/0:0 in3:05, zero fits/updates. Same100 test rows,
three saved states,21 cell traces; native/manual h/c arithmetic and saved
sequences match exactly. Finalp00 GRU8 candidates are allzero fromstep4;
z stays near.5, so retained state decays to5.92e-18. Largest final-unit
witness has POSITIVE bias; increasingly negative input projection closes it.
Do not generalize the earlier negative-bias explanation. Finalp30 LSTMs
admit large positive recurrent candidate injection with open gates; growth
starts inencoder1 and reaches decoder3~1.39e15. These are fixed-weight
forward mechanisms, not training-onset or successful-repair evidence.
All45 copiedfiles,15 pairfiles,10 prior diagnosticfiles and all parameter/
optimizer hashes verify; local audit matches clusterbytes. Stop this check.
Next define one source-explicit recurrent-cell alternative/control and require
full48-step constructed learning before any new CER fit; no branch/job is
selected or launched. Website unchanged. See [gate record](../studies/takiddin-2021-robust-poisoning/results/sequential_gates_20261002/README.md).

Raw local/Panther output:sequential-gates-20261002-attempt1. Frozen remote
checkout /export/home/fjoad/atk-evidence-seq-gates-20261002. Resultrecord
stores raw JSON losslessly compressed plus compact summary/PNG/SVG; original
uncompressed JSON and NPZ traces remain under data/derived. Research replay
56.13s; wholeallocation185s includes checks/audit. Four gatecontrols pass on
CPU/GPU;403 repository passes/41 skips, strictdata and journal checks pass.
Five direct sciencefiles unchanged. No local research model calls.

**Earlier gate authorization checkpoint:** the user approved a zero-fit check of the
saved recurrent operations. Same first100 test rows, shared initial/finalp00/
finalp30, all six LSTM cells and GRU8. No parameter update or repaired state
trajectory. Four constructed controls pass;403 repository cases pass with
41 environment skips. Freeze and submit one10min V10016GB/4CPU/16GiB job,
7min internal guard, then copy/audit/report and stop. No job yet at this
checkpoint. Plan:2026-10-02-sequential-gate-arithmetic.md. Website unchanged.

**Saved-state diagnostic complete:** frozen3127b19372cd16546cbf814394ea8d52f7ee2f28,
Panther408550 COMPLETED/0:0 in2:11, zero fits/optimizer updates. On fixed100
train/test rows: initial logit differences survive but float32 probabilities
round together; finalp00 GRU8 is about6e-18 and classifier hidden profiles
are identical; finalp30 encoder/decoder reach3e9/1e15, bridge is exactlyzero
fromstep2, and GRU2 onward is profile-identical. Upstream gradients are tiny
but nonzero; no whole native stage is entirelyzero. Widening only the final
readout does not recover either trained model. All10 copiedfiles,15 oldfiles,
20 preparedarrays, all model/optimizer hashes, native decoder replay and
savedtest parity verify; local audit matches clusterbytes. Endpoint/batch
scope only, not a training trajectory or family-wide impossibility. Next
consider saved gate/cell arithmetic before declaring a repair; no furtherjob,
fit/seed or website work. See [diagnostic](../studies/takiddin-2021-robust-poisoning/results/sequential_saved_state_20261002/README.md).

Remote code /export/home/fjoad/atk-evidence-seq-observe-20261002 remains frozen.
Raw local/Panther output:sequential-saved-state-20261002-attempt1 under the
original study data/derived path. Log and accounting copied into resultrecord.
Six local/cluster observerchecks pass;403 repository passes in recordedrun;
strictdata/journalchecks pass. No local research inference. Original pair407294
and five direct scientific files unchanged. Next unresolved question is exact
recurrent gate/cell arithmetic, not an automatic higher-precision repair.

**Earlier October2 authorization checkpoint:** first100 train/test rows,
sharedinitial/finalp00/finalp30, zero training/optimizer updates. Observe native
AEA/GRU/head outputs, gradients and bridge/affine readout arithmetic; weights
and originals must remain unchanged. Six discriminating constructed controls
pass locally in3.55s. One10min V10016GB/4CPU/16GiB allocation with7min guard;
no job yet. Freeze then inspect queue/output before submission; run onPanther.
Plan2026-10-02-sequential-saved-state-diagnostic.md. Local pinned fixture
runtime:tmp/robust-feedforward-venv/bin/python; normal tests:.venv/bin/python.
No direct scientific model edits, new fitted branch or website work.

**Sequential pair complete (October1):** scientific3705bcca04279bbfa8523a573b9b573f123463b7;
job407294 COMPLETED/0:0 in1:21:28 onPanther005,oneV10016GB,4CPU/16GiB.
Both50epochs/2250updates, paired initial arrays and original20-customer data.
Scalar-Sigmoid interpretation; test probabilities constant0.5047591328620911
and0.3539589047431946. DR/FA100/100 and0/0,AUC50/50; all-cutoff/reversal DR0
at declaredcaps. FinalBCE0.693111/0.650009 near constant-prior0.693094/0.649853.
Initial probabilities also constant. P00 intermediate retains profile variation
but p30 nearlyzero; itsMSE equals zero baseline, not useful reconstruction.
All15 copiedfiles hashmatch, local pair/comparison byte-match, HDF5weights/
config/optimizer2250/norm and compute-node reload checks pass. No local model
inference. Stop pair; next propose zero-fit activation/gradient inspection of
saved initial/final48-step models before more training. No newjob/fits/seeds.
Preflight407255 completed0:0 in8:01,14checkspass, warmprojection1845.84s<=3600;
SHA721256059d34a104fa2eda04ccb0bb8775603d01206117e4e82e1a9e54782c07.
Pair was moved while pending from2h-limitedgpu-short togpu-all; samejob/3h
ceiling/resources/science, no duplicate. Wholefile login hash attempt killed;
streaminghash succeeded. Archiveaudit ignores only shared_object_id metadata.
Original results unchanged. Local400tests passed/34skipped atlaunch; strictdata/
journal checks passed. Alllargework onPanther, small localfixtures permitted.
No website changes/publication. Records:results/sequential_pilot_20261001 and
sequential_preflight_20261001. Raw pairdata exists locally and under original
Panther atk-evidence/data/derived/takiddin-2021-robust-poisoning/sequential-sigmoid-pilot-20261001-attempt1.
Isolated executing checkout /export/home/fjoad/atk-evidence-paper3-seq-20261001.

**Oct1 interface comparison:** source pages2678/2680/2682/2683 checked;
readout shape/equation still omitted.73fe054 froze primaryI-SEQ-scalar-sigmoid
from component-output reading and C-linear control BEFORE results. Four
full-width8step synthetic fits300updates all100%/BCE~1e-7, identical initial
weights to historicalReLU. Artifact/fresh reload/metrics pass, oldReLU default
and attention outputs preserved. Sigmoid alone was preselected for numerical
continuation. models.py serializes bridge_activation, ReLU default unchanged.
Direct fit_sequential/CLI TableV/audit +20min GPU preflight and180min paired
launcher now implemented. Preliminary scripts/tests pass; complete freeze
checks before submit. Same scientific models across learning/timing/fit; both
50epoch/batch100 fits have4200s guards. Full48step GPU time not measured yet.
Panther authenticated; idleV10016GB listed. No job submitted at this checkpoint.
No actual electric-data fit/scoring or site change. All old outcomes retained.
See SEQUENTIAL_INTERFACE.md, results/sequential_interface_20261001 and
docs/plans/2026-10-01-sequential-interface.md. The learning controls were local
software fixtures; user reaffirmed small local checks are fine and big jobs
must run on Panther. Use existing pinned remote neural environment.

**Full-width learning gate failed (September28):** User approved all bounded
dependent stages, not a waiver of failed checks.66db165 froze width-only
fixture: same32train/32test,8steps, seed20260920, Adam.001,300updates,
600s/fit guard; published widths9,240,802 params. Bothnormal/reversed finish
105.63/103.80s at50%, BCElog(2), p=.5 everywhere. Initial/final weight hashes
match across orientations; initial/final gradients zero. Initial decoder
features active but scalarReLU rawprojection negative (train -0.02035 to
-0.001265), so projected data/allGRUs/head are zero. This scalar bridge and
activation are OUR recorded source completion. Do not call it a fully printed
operation or paper-data failure. Fresh reload and every artifact/source/data
hash/metric pass; old files unchanged.397tests pass/30skips, strictdata and
journalchecks pass. No GPU/research job or scoring/site work; staged sequence
held. Next source review: decoder-to-GRU tensor shape/activation, then separate
declared alternative/control. No automatic seed/activation retry. Records:
results/sequential_full_width_20260928; plan2026-09-28-sequential-execution.
Panther reached authentication in a batch probe, but no remote job submitted;
scientific gate failure, not access, stopped the dependent stages.

**Constructed collapse trace complete:** User approved tracing the failed
reversed fit.9368a90 froze one50-update replay, same small model/data/seed/
Adam.12.37s CPU; all50 old losses and initial weights match exactly;
observer preserves state. GRU8 and classifierhidden both become zero after
update2 on32train/32test and stay zero through50; upstream feature gradients
zero. EarlierAEA/GRU1-7 active. Classifier head updatedparams kill oldinputs;
oldbias restores local activation. Separately declared read-only GRU8 swaps
show updatedcandidatebias alone kills oldinputs/params; oldcandidatebias
restores some activation with updatedinputs/params. Native/manual GRU agree
within8.93e-12. All51snapshots and oldartifacthashes verify; fresh0/1/2/50
outputs exact. No repaired training result or infinite-time exclusion.
Gate stays failed. Next bounded question: full-width constructed learning
with existingoptimizer to assess miniature-fixture artifact, not settings/
seed search. No GPU/research job, real-data score/fit or website change.
397repo passes/30skips;3new checker tests; strictdata/journalcheck pass.
See results/sequential_trace_20260927 and its September27 step plan.

**Sequential implementation/constructed gate:** User approved continuation.
Model added to direct models.py atb9d8223: full9,240,802 parameters, verified
forward shapes. All9 software/gate checks and6 oldAEA checks pass. Fixed
small model5,082 parameters (8/6/4 LSTM,8x8 GRU,Dense16,8steps),32train/32test,
seeds20260920/20260926,300updates each. Normal100%/BCE.22528521; reversed
50%/BCElog(2). Same initial weights; both complete. Reversed finalGRU and
classifierhidden zero on toytest; intermediateAEA positive; upstream final
gradients zero. Complement-normal probabilities solve reversed labels: capacity
exists, training did not find it. Not evidence on full-width training or CER.
Normal saved model/history before JSONwriter np.bool_ failure.2f9b793 fixes
writer; recovered normal report without refit and ran only missing reversed
case. Original attempt1 and attempt1-recovery preserved underdata/derived.
Result/audit/logs/recovery scripts inresults/sequential_constructed_20260927.
Two read-only audit-hook/output-shape errors also recorded; no model changes.
GPU timing held by failed learning gate; no allocation/researchfit/scoring.
Next bounded question: diagnose reversed training collapse/reduced-width
instrument. No seed/activation search. Website unchanged.394repo passes/30
skips;15pinned checks, strictdata and journalcheck pass. See September27 plan.

**Next-step source/design (September26):** User asked to perform source
resolution and define the bounded next experiment. All10 target pages read
and visually checked; ref23 pp425/426/428/429 checked. New
`SEQUENTIAL_ENSEMBLE_PILOT.md` proposes I-SEQ-native-IVC: joint correctedBCE,
original two-class p00/p30, ReLU/Adam/dropout0/MaxNorm1 perIV-C, scalar final
Sigmoid score and declared0.5 cutoff. Native cells and intermediateReLU
scalar projection are explicit completions; alternative activations remain
open. Standalone AEA MSE/MAE/0.51 do not select ensemble settings. No
standalone repair promotion or pretraining prerequisite. Proposed architecture
count9,240,802 is arithmetic only. Next implement/verify ensemble with
positive learning, causal feedback/query and fresh-reload checks, then full
variable-batch GPU preflight. Proposed20min preflight+180min pair (70min/fit)
not executed/approved by the document alone; research checkpoint remains.
Existing GRU is contextual, not a matched AEA ablation. Broad statistical
attainability and causal reconstruction claims remain untested. No new
research scoring/fit/job, direct-file or website change. Plan
`docs/plans/2026-09-26-aea-next-experiment.md`.

**AEA recheck (user requested):** MAE promotion withdrawn; no training branch
selected. Its optimistic free-cutoff pass was confused with preserving 0.51;
fixed-cutoff minimum FA is 36.75239/33.79187%, above 5.25/18.45%. Necessary
free cutoff intervals [0.78932,0.97132)/[0.60363,1.25402). Source axis is
unspecified; score norm does not choose training loss. Repair README p30
bounds used the wrong FA cap: correct MSE/raw/global DR bounds at 18.45% are
100/100/96.81548%. Raw JSON unchanged. Original audits replay exactly;
independent endpoint/pair-count/order-statistic calculations verify all
branches, all-cutoff/fixed-cutoff distinctions and original geometry. Original
model/data artifacts and five files preserved. Old prototype query test also
passed with query disabled; save succeeded but load failed. Fixed module-scope
registered layer, explicit seed config/loss argument and six constructed TF
tests (causal query witness with negative control, fresh-process reload).
24 weight arrays/output/attention equal original small seed42 fixture exactly.
No useful-learning claim, research fit/job or website change. See
results/aea_recheck_20260925 and docs/plans/2026-09-25-aea-recheck.md. Before
any new experiment separately justify scaling, loss, score and cutoff; no
automatic alternate branch or model search.

**AEA repair envelope complete:** job403408 failed prelaunch in1s from a
misspelled output directory; no data loaded. Corrected job403409 completed0:0
in41s oncrimv3srv024, requested1CPU/8GiB/noGPU, Slurm2logicalCPUs, threads1.
Five interval/oracle branches, zero fits;56inputarrays/10archives. Feature-z
MSE oracle excludes p00 paper corner: upperDR88.24405 versus94.05 rounding-
favorable target; global-z MSE upperDR92.26190 also excludes p00.
Feature-z MAE/raw-unit MSE/minmax control pass the free-cutoff envelope; this
means only not excluded, not trained or realizable. Common p00 cutoff interval
empty for feature-z MSE/global-z MSE. Prior geometry unchanged, local audit
matches cluster, result SHA fd14a718560989e5575e7e1a8f4f5f153a17fc6fd023ccfdde3ee35ade9c7f69.
The recheck above corrects the later MAE promotion. No neural fit or branch
search. See results/aea_repair_20260925; website untouched.

**Repair envelope approved:** user "continue", then resumed interrupted turn;
interruption only completed connectivity, no code/job at that time. Now
AEA_REPAIR_ENVELOPE.md and checks/aea_repair.py/run_aea_repair.sbatch specify
five zero-fit branches: feature_z_mse,feature_z_mae,raw_unit_mse,
global_z_mse,minmax_mse. Last is C control; no test-statistic fitting/clipping.
Oracle gives benign rows lower error and attacks upper error; hence ROC/AUC
upper bound for every reconstruction in those intervals. This is NOT the old
clipped-score AUC and NOT a model. Tests include identical-input contradiction
where oracle passes but a deterministic model cannot. Strict>cutoffs and
ties, nominal/favorable rounding+1e-6 guards, necessary cutoff/factor intervals,
original/synthetic strata, primary-only reversal; linear-output case analytic
only.22 fixtures pass. No real repair calculations/job yet; finish main checks,
freeze then submit one10min/1CPU/8GiB/noGPU job. Prior result d6e8b877...a42e5
and56 inputs must verify first. Plan2026-09-24-aea-repair-envelope. No website work.

**AEA geometry complete:**5d5d85050850f21f344c7509267b8215ca18767d,
job402811 COMPLETED/0:0 in30s, nodecrimv3srv025. Requested1CPU/8GiB/noGPU,
Slurm allocated2logicalCPUs, CPUs/Task1/numericalthreads1. Program1.80581s,
peakRSS204144KiB vs coarse Slurm51316KiB. Zero fits/inference/regeneration.
For fixed novelty inputs/[0,1] reconstruction/MSE at printed0.51, minimum
FA22.24880/22.78708 literal; allowing tau.515+1e-6 and printedFA rounding,
741/3344=22.15909% and748/3344=22.36842% must be false alarms, beyond
5.25/18.45 allowed. Any weights/seed/training duration with those fixed
semantics cannot rescue that point. Not a full-population/scaler/threshold/
other-output/other-error/ensemble result. Favorable RMSE minima55.14354/
53.70813; SSE99.52153/99.55144. Optimistic DR bound100%, not excluded.
Original-benign MSE minima53/187=28.34225,62/187=33.15508: not synthetic-only.
Controls AUC p00/p30: zero51.86267/48.65709, half60.86976/58.98843,
clipped59.44402/56.65526, dailymean65.70123/65.70123. These are fixed
baselines, NOT model-AUC limits. Confirmed373train/6704test,108contaminated
p30 rows and108exacttrain/test overlaps; all56hashes/scaler transforms/
synthetic parents/rawtestpair/scorebounds/metrics pass. Local audit matches
cluster bytes. Original inputs unchanged; raw score archives stay ignored.
ResultSHA d6e8b877ce31bd6ccaccbfa9ed478fad9587665782a9192590243f58834a42e5.
Results/aea_geometry_20260924; ignored aea-geometry-20260924-attempt1.
Stop. Proposed next: bounded score/scale clarification/repair comparison,
not fit excluded0.51 configuration or add seeds. No further job/site change.
Contract aa2c4ab unchanged; implementation root CPU environment reused.

**AEA source step complete:** user approved specification, not research fits.
All10 target pages re-read/rendered; Zhao et al.[23] pp425–430 read/visually
checked from Augsburg PDF (ignored references/zhao-2020-hierarchical-attention-autoencoders.pdf,
SHA eacc7bc72db20c13b7a4f66de0837900ba5204e479f09c3a842ea73b9928b525).
Sun/Wu[22] full text not located; do not claim to have read it. Source map and
proposed I-AEA-native-table are in AEA_SPECIFICATION.md: standardized48x1,
encoder500/300/200 mirrored decoder200/300/500, nativeSigmoid LSTM nopeepholes,
causal decoder-query attention, zero initial output/free-running decode,
Sigmoid scalar reconstruction/MSE, SGD.01, MaxNorm1; all omitted choices explicit.
This model is NOT implemented or fitted. Printed peepholes/tanh and attention
loop/state indexing differ or are incomplete. Ref23 is not imported wholesale.
Range mismatch yields pointwise MSE bounds, not automatic detection failure.
Eleven constructed tests pass; classification-only complement witness is not
an ensemble experiment. No CER arrays scored or cluster job submitted.
Novelty metadata:373train/6704test, p30 replaces108rows from6customers;
108 exact train/test row overlaps,373 sharedsource days. Test3344benign
(187original/3157synthetic),3360attacks. Scaler changes by poison condition;
do not use classifier2232-row inputs or claim matched scaled features.
Next proposed AEA_GEOMETRY_CHECK.md: zero-fit range/threshold/baseline and
input/provenance check,10min/1CPU/8GiB/noGPU, separate approval before execution.
Its runner is not implemented. Five reproduction files and all results unchanged;
website belongs to another session. Plan docs/plans/2026-09-24-aea-source-specification.md.
Final371pass/17skips;17focused source/geometry tests pass; strict data passes.
An attempted METHOD annotation triggered its legacy hash guard and was removed;
METHOD/oldaudit/test remain unchanged. New notes live only in AEA_SPECIFICATION.

**GRU complete and audited:** 1f847039ea3c5d410db5e9ee194a6625ecd2b54d,
job 402625 COMPLETED/0:0 in 56:11, crimv3mgpu005, one V100-16GB/4 CPUs/
16 GiB. Both 50 epochs/2,250 updates, no guard reached. Initial weights match
each other and old partial; first 32 p00 epoch losses/accuracies/updates match
old exactly. DR/FA/AUC p00 62.35294/9.93789/89.07390; p30
39.72851/6.12245/79.88605. At corresponding FA caps 6.8/20.6%, best DR
51.40271/67.14932 versus 92.4/78.5; reversal .27149/6.24434. Poisoned AUC
close to 79.4 reported, not whole-row reproduction. Common-cap DR declines
10.04525/16.56109/16.56109/12.48869 at 6.8/17.6/20.6/33.3; AUC drop9.18785.
P00 loss .37406 at37→.58846 at40→.44132 at50; p30 final .43377. No plateau
or global limit. Final observed training accuracy76.56810/73.40950.
Fits1316.25543/1375.62402s; program1513.03924/1580.91651s. All GRU jobs
including old preflight/partial total4889s=81:29; do not erase repeated cost.
All20 input arrays, IDs/labels/hashes, scores/metrics, pairing/schedule/budget
pass. Local pair/comparison/prefix audits match cluster bytes; HDF5 final
weights, optimizer2250,4058702params, configs/norms verify without inference.
All artifacts in ignored gru-completion-20260924-attempt1 under usual derived
study root; summaries in results/gru_completion_20260924. Result SHA p00
ce6847aba5c8c5da942779c02b100272ae7b0fa4e21e9eec9e2b0d52a05b4f9b; p30
c4de02d467b0c584477a93a7ea7130e9b474e4989bfdefbd1222a231d6817a26.
Remote checkout /export/home/fjoad/atk-evidence-paper3-gru-completion-20260924;
log /export/home/fjoad/robust-gru-completion-transfer-OceGkq/slurm-402625.out.
Stop pair. Proposed next: source-specify standalone AEA/reconstruction, not
extra GRU seeds/epochs or ensemble fit. No new job/inference/site publication.
Old partial remains intact. This supersedes all running-state text below.

**Approved GRU completion:** user said go ahead after the partial-audit report.
docs/plans/2026-09-24-gru-completion.md and GRU_COMPLETION.md authorize one
same-seed original p00/p30 restart, unchanged50epochs/batch100/model/data,
35min guards/85min allocation, one V100-16GB/4CPUs/16GiB. Budget is
ceil((50*max32full_epoch_seconds)/300)*300=2100s; total2*2100+900=5100s.
Constructed fixture confirms old timing fixed100 batch versus researchNone
batch and64-row remainder; timing cause not isolated. No extra GPU timing job.
Keep original900s default, old720s false gate and original outputs unchanged.
Fresh initialization rather than pretending unsaved shuffle/dropout state
continues exactly. Compare newp00 initial hash and first32 loss/accuracy/update
records with old; differences must be reported, not used to select a run.
New CLI only exposes guard900/2100 and records budget/contract. No job yet;
finish tests/freeze, then scheduler/output check and one submission. No website work.

**GRU stopped and audited September 24:** job 402378 FAILED/2:0 in 22:45
(intentional partial-fit return). P00 hit 900.07785s guard at 32 full epochs
+7 batches of epoch33, 1447 updates. P30 never started; no pair audit exists.
Partial DR42.98643 FA8.78438 AUC80.82379; best DR35.74661 at FA<=6.8 versus
92.4 printed; reversal1.44796. No cutoff rescue for this partial model, not
a completed50epoch limit. Loss .67070→.40140 across full epochs; not plateaued.
Real full epochs19–32 took24.72–25.90s, slower than constructed estimate.
Scoring135.86020s/reload47.37145s; process1096.92745s. No OOM; TF peak903443968B.
Read-only audit verifies 10 consumedarrays/source/outputhashes/labels/scores/
metrics/initial-finalweights/1447 serializedoptimizerupdates/config/norms.
No local model inference; recorded GPU reload verified by original result hash.
Raw result SHA fa19293a1bac847fd7fb3ab8fe37b25f3ea92f52d88ad8e53cee129bbbdd4602.
Full artifacts copied to ignored data/derived/.../gru-pilot-20260923-attempt1;
small result/history/log/audit in results/gru_pilot_20260923. Ordinary completed-
fit auditor still refuses partial. No active account jobs or new submissions.
Stop this attempt. Next proposed question: timing calibration/adequate budget
for unchanged50epochs; explicit approval before new compute. Saved optimizer
does not by itself preserve unsaved shuffle/dropout state for exact continuation.
Website untouched. This supersedes all running statements below.

**Live GRU allocation:** job 402378 submitted ONCE and running on
crimv3mgpu005; launcher 46966c7e021e964ff582e1eb56422b75cc6a4ec7,
science unchanged at 2d706b103ee03cc705cc3ef3f07718bc3ed7792a. One V100-16GB,
4 CPUs/16GiB, 40min job/15min fit guards. Remote checkout
/export/home/fjoad/atk-evidence-paper3-gru-approved-20260923; output under
the usual derived study root at gru-pilot-20260923-attempt1. Log
/export/home/fjoad/robust-gru-resume-transfer-jfMFay/slurm-402378.out.
The cluster verified the exception/current scientific sources; authorization
matches local bytes. All 16 fixtures passed in 100.295s. P00 real-data training
has completed at least its first full epoch/45 updates with finite loss and
saved initial weights/history. No final performance is available yet.
Latest observation: second epoch completed in 29.64104s, 90 updates, finite
loss .63577533. Slower than constructed timing; if sustained, 15min guard
will give a partial fit and stop before p30. Do not silently extend the guard.
Check scheduler/output before doing anything; do not duplicate or rerun.
The wrapper handles p00 then p30 then analysis/audit, stops on partial/failure.
No completed research result at this checkpoint. See results/gru_pilot_20260923.
Local checks: 16 GRU+8 FF passed; main 355 pass/17 skips; strict data passes.
No website work. This live checkpoint supersedes all unsubmitted statements.

**Authorized resume:** user replied “sure incresae it and keep going” to the
explicit GRU runtime-gate exception question. Only the launch ceiling is now
900s; the original 720s failed gate is preserved, not relabeled a pass. Retain
40min job/15min fit guards and unchanged scientific code/inputs/settings.
GRU_RUNTIME_EXCEPTION.md binds the preflight SHA/revision below; verifier
checks historical source and current scientific-file equality. Submit only
the p00/p30 pair after checks, preserve partial outcomes, audit and stop.
No website changes. This supersedes the awaiting-user statements below.

**GRU checkpoint:** frozen 2d706b103ee03cc705cc3ef3f07718bc3ed7792a;
constructed GPU preflight job 402376 completed 0:0 in 153s, V100-PCIE-16GB,
4 CPUs/16 GiB, node crimv3mgpu005. 13 fixtures pass locally/on GPU. Twelve
constructed updates (two warmups), warm range .33034-.34608s, mean .33672s.
Slowest-step fit projection 778.68199s exceeds 720s gate: correctly held, NOT
a paper-performance failure. No CER input loaded/research fit/pair submitted.
User asked for explicit gate exception within unchanged 40min pair/15min fit
guards; no reply at this checkpoint. Do not launch until decision. Source/
transfer/device/norm audit passes. Result hash 238384ed7e52f1438a9bf66202e850b1afa3b18215019dc8e284021633a3a040.
Program 26.4718s, eager inference 100-row batch 2.118s, TF allocator peak 855042560
bytes (not whole VRAM), process RSS 1230216 KiB; Slurm max 1442064 KiB includes tests.
Records results/gru_preflight_20260923; ignored gru-preflight-20260923-attempt1.
Main tests 352 pass/17 skips; neural 21 pass, Ada 7 pass. Old FF audit byte-identical.
No website files/journal/publication changed; another session owns that work.

**Current scientific work:** user approved the GRU-first continuation toward
AEA/ARIMA/ensembles; website belongs to another session. GRU_PILOT.md fixes
I-GRU-native-table: 8x300 GRU, 48 time steps x1, ReLU candidate, sigmoid gates,
reset_after=False, input dropout .2/recurrent 0, MaxNorm 5 both kernels/output,
Dense 2 Softmax, categorical CE repair, Adam .001, float32, 4,058,702 parameters.
Reference [26] arXiv 1809.01774 all six pages read/rendered; explicit two outputs
and corrected log(1-p), but generic tanh/V-softmax equations remain ambiguous.
Reference source PDF is ignored; hash 253568ec...d749fd4. The target's selected
settings take precedence; a one-step/48-feature input remains an unrun branch.
13 GRU and 8 FF neural fixtures pass; old FF audit matches bytes. Constructed
timing uses model.fit, not repeated train_on_batch (retracing observed locally).
Before any research fit: freeze code and run only the ten-minute constructed
GPU preflight; gate 2250*max(ten warmed update times)<=720s. If it passes,
one 40-minute V100 pair at original p00/p30, seed 20260920, 50 epochs/batch 100,
15-minute batch guards. No real-data work locally, no extra seeds/settings.
Panther authenticated; neural env and idle V100s available; no GRU jobs yet.
See docs/plans/2026-09-23-robust-gru.md. Do not regenerate/publish website files.

**Latest navigation preference:** the website must be open-ended. Remove the
cross-paper tabs and “Paper N of 3” labels; each article links back to the
growing homepage index. No fixed collection size in headings, footers or
metadata; homepage entries are unnumbered. Research contents stay unchanged.
See docs/plans/2026-09-23-open-paper-index.md.
Published `616a0c0`; Pages 35854458279 and CI 35854458323 passed, with 32
focused local tests. Live index and article verified: no tabs/fixed counts.
Notebook stylesheet URLs are content-versioned to avoid stale cached layouts.

**Latest website preference:** user wants a continuation, without prominent
dates. Main pages now use descriptive section headings, undated introductions
and no update-date banners. Internal log dates and published anchor IDs remain
for provenance/link compatibility; paper publication years are retained.
No scientific content or archived records change. Follow-up plan:
docs/plans/2026-09-23-continuous-website.md.
Published at 3cd566e; Pages 35852601219 and CI 35852601197 passed. The 31
focused tests and live-page verification passed. Continue using this undated
presentation rather than restoring calendar-based section headings.

**Website/user direction, September 23:** user approved a quiet index linking
the three registered papers, notebook pages with claims → starting hypothesis
→ chronological investigation/results/corrections → current conclusion, and
publication after checks. No grand headline card or cross-paper campaign.
New `scripts/render_journals.py` renders all three RESEARCH_LOG.md sources;
old command is a compatibility wrapper. New pages use `site/notebook.css`.
Paper 1/2 earlier root pages are preserved as `earlier-notes.html`; detailed
Paper 1 `/reproduction/` stays available. Source/result bytes unchanged.
User clarified older water and deep-autoencoder work used 5.6 Sol and earlier
thinking; rethink/redo those later, one paper at a time, preserving results
for now. Their notebook notices explicitly say awaiting reassessment, not
fresh validation. Paper 3 adds mathematical issues with aggregation caveats;
main sequential ensemble remains untested. Full suite 344 pass/10 environment
skips; 30 site/registry tests pass, mobile/desktop browsing checked. Published
9f21c28a44382275435e967efced1ead2ae6fcb7; Pages run 35850178097 and CI run
35850178039 succeeded. Live homepage/all three tabs and key caveats verified.
Earlier HTML content is preserved apart from archive comments/back links;
the research result bytes and source contracts are unchanged. No new scientific
run. See docs/plans/2026-09-23-notebook-website.md.

**Current order-control result, complete September 23:** contract 9b23b32,
startup-only repair 579a3f5cafc0f8943f230ff7458a1c14103a6718, CPU job 402291,
0:0 in 20s on crimv3srv024, 4 CPUs/16 GiB/no GPU. Exactly one new forest;
saved B-p00/B-p30 not refitted. D-p30 versus B-p30 on 1,288 original rows:
DR 63.16742 vs 63.89140, FA 12.56831 vs 8.74317, AUC 79.80343 vs 84.37752.
At common FA<=17.6/33.3%, DR 68.32579/75.56561 vs 75.65611/85.42986.
Observed attack prior 34.86320→48.72032% but no default DR rescue; AUC drops
4.57409 points. Whole resampling-policy contrast, not isolated prior causation.
Useful signal remains above daily-mean AUC 65.43382. Same 2,632 originals,
675 flips/six customers; D has 611 synths/3,243 total, observed 1,663/1,580.
Parent truth categories 0/1/2 attacks: 157/132/322, so 454 unknown synthetic
truths remain training-only; true-label training accuracy null. Observed
training accuracy 99.87666%. All 28 zero-poison arrays equal B-p00 exactly;
28 new+56 reference arrays, ancestry/scaling/labels/settings/reload/metrics
pass. Local audit byte-identical to cluster. Fit 0.74316s, program 4.75547s,
process peak 242,148 KiB; Slurm coarse sample 3,388 KiB is not the peak.
Result SHA e65e397c387263299f0f0e451d5e7c5d1640dbf969c6fa64965bebf6d1004aea.
Initial job 402290/frozen 9b23b32 failed after 57s in NumPy version alias lookup,
before input load/output creation/fit. Log preserved, only lookup repaired.
Resume requested 14:03 but scheduler stored 15:00; actual combined 77s, not an
enforced cumulative 15-minute hard cap. 339 tests pass/10 environment skips;
12 targeted fixtures pass locally/on compute. See results/poison_balance_20260923
and ignored poison-balance-20260923-attempt2. Stop this diagnostic. Proposed
next coverage step: separately specify GRU (shape/activations/output/loss/defaults)
before implementation or fit. No new model/seed or publication authorized here.
Original primary files/results unchanged; journal/site draft stays local.

**Current feed-forward pair, complete September 22:** VPN/SSH restored;
setup job 398992 completed 0:0 in 12:54, not rerun. Frozen b5da23a ran as
GPU job 400825, completed 0:0 in 3:28 on V100-PCIE-16GB (driver 570.133.07),
4 CPUs/16 GiB, within 20 minutes. Both repaired-BCE fits completed 50 epochs,
batch 100; 2,250 updates, identical initial weight hash a86bd4ec...bc0d74;
same original p00/p30 arrays and 675 changed labels. DR/FA/AUC
89.14027/7.45342/96.35480 and 51.58371/0.53239/90.89064. At FA<=9.3%,
p00 DR 90.76923 rounds to 90.8 printed; at FA<=24.4%, p30 DR 88.86878 exceeds
76.0. Default DR drop 37.55656 versus common 9.3/24.4-cap drops 13.30317/
6.60633; AUC drop 5.46415. No full-row reproduction or population inference.
All 50 histories/updates, model reloads, paired weights, 20 consumed input arrays,
labels/IDs, hashes and metrics pass; local/cluster comparison AND audit files
are byte-identical. All 8 neural fixtures passed on GPU. Fits 18.284/17.763s,
TF allocator peaks 28.30/28.49 MB (not total VRAM); no full-data timing claim.
Source: six 500-ReLU layers→1 Sigmoid, Adamax 0.002, MaxNorm 3 axis 0, no dropout,
seed 20260920, float32, explicit BCE repair. TF 2.16.2/Keras 3.4.1 pinned env.
Local interpreter tmp/robust-feedforward-venv/bin/python; Panther env remains
/export/home/fjoad/atk-evidence/data/environments/robust-feed-forward-2.16.2.
See results/feed_forward_pilot_20260922; result hashes f28ef620...e3c3c7b3
and 2256d0ab...0a689861. Original models/arrays untouched; all weights/probabilities
remain ignored. Stop the pair. Next proposed question: shared poisoning versus
balancing order and changed observed class proportions, before any controlled
run. No new experiment is approved by this summary. Other models and SVM
sensitivity remain open. Journal/site draft updated locally, not published.

**Current read-only SVM follow-up, complete:** frozen 4665e07, job 398978,
completed 0:0 in 32 seconds; program 5.61120 seconds, zero experimental fits.
Manual/libsvm replay on 13,392 train/test row-model pairs: largest error
1.67688e-12, no prediction disagreement or near-zero ambiguity, native test
scores exactly match originals. Support vectors/coefficients/labels bind
correctly; all original input/model/score hashes unchanged. Fixed 512-row subset
(UID rule, seed20260921/role601) has 309 originals/203 synthetics; observed
benign/attack251/261 at p00,341/171 at p30. K min=-23.45098; centered min=
-19.96835; both have366 resolved negative eigenvalues at about 1.5e-8 tolerance.
Centered witness obeys both dual equalities (residual 2.05e-15), so the usual
concave-dual guarantee does not apply here. Do NOT infer global suboptimality,
KKT failure, a feasible improvement at the fitted box boundary, caused accuracy
loss, or all-parameter impossibility. Three archives/witnesses audited locally;
audit matches cluster bytes, old SVM audit still matches. Requested 1 CPU,
8 GiB,10 min; scheduler allocated 2 logical CPUs on a two-thread/core node;
CPUs/Task=1 and numerical threads=1. Peak process RSS215816 KiB. No GPU.
325 tests passed before freeze,4 Ada fits skipped there and passed separately;
11 diagnostic tests pass locally/on compute. Result SHA cdc43fa5...7918013.
See results/svm_replay_20260921. Stop this diagnostic. Next proposed coverage
step: specify feed-forward baseline and bounded pilot with standard BCE repair
explicit; finite SVM gamma/coef0 sensitivity remains open. No new detector
fit, parameter search, or publication occurred. Local journal/site updated.

**Current SVM step, completed September 21:** user-approved original p00/p30
pair frozen e698173, job 398709, completed 0:0 in 2:00 under 15 minutes,
4 CPUs, 16 GiB, no GPU. C=1, sigmoid, gamma='scale', coef0=0, no probability
calibration, tol=.001, seed 20260920. Raw decision scores are preserved without
probability transformation; native predictions agree (zero tie maps to 1;
no actual test ties at 0). DR/FA/AUC 62.08145/39.39663/65.64194 at p00 and
39.45701/31.14463/63.38126 at p30. At corresponding paper FA caps 10.2/25.7%,
best saved-score DR 19.36652/33.48416 versus 89.2/73.7; reversal does not help.
All 2,224 score boundaries/model inspected. Both fit_status=0, iterations
1168/1115, support vectors 1689/1901; observed-label train ACC 63.33/59.72.
Fits 5.019/5.106 sec. Gamma 0.020833333524383626 versus auto 1/48 differs
relatively 9.17e-9, not proof of identical refits. Simple daily-mean AUC 66.19624.
All 20 consumed arrays, hashes, 675 flips, paired features/IDs/labels, raw-score
reload, and metrics pass; local comparison AND pair-audit match cluster bytes.
314 main-suite tests pass, 4 Ada fit tests skipped there and passed in their
isolated env; 7 SVM tests pass locally/on compute. An old source-audit test
was repaired to verify original/corrected METHOD documents at their explicit
Git revisions; arithmetic and historical evidence remain unchanged.
See results/svm_pilot_20260921. Stop the pair; next proposed question is a
bounded read-only score replay/kernel diagnostic with fixed sampling and
tolerances, not new seeds/fits. Kernel spectrum and cause remain unmeasured.
No alternate kernel/gamma/coef0/calibration/preparation or next model ran.
requirements-svm.txt pins sklearn 1.9.0, numpy 2.5.1, scipy 1.18.0, joblib 1.5.3,
threadpoolctl 3.6.0. Expected parameter/pickle deprecations retained; no actual
fit convergence warning. Journal/site draft updated, not deployed.

**Current AdaBoost step, completed:** user-approved pair frozen at d47a6de,
job 398348, completed 0:0 in 14 seconds (15-minute cap, 4 CPUs, 16 GiB,
no GPU). ADABOOST_PILOT.md uses original p00/p30 arrays and stock sklearn
1.5.2 SAMME.R, learning rate 1, seed 20260920, depth-one trees. Both fits
used all 50, taking 1.494/1.503 seconds; exact reload passed. NumPy 1.26.4,
SciPy 1.13.1, joblib 1.4.2, threadpoolctl 3.5.0 isolated from the RF env.
DR/FA/AUC 81.08597/15.17303/90.70692 at p00 and
46.42534/5.05768/83.73956 at p30. At FA caps 14.1/29.9%, best DR is
80.72398/88.77828 at p00, 67.33032/80.45249 at p30. Default DR drops
34.66063 points but AUC drops 6.96736; thresholds explain part, not all.
The poisoned fixed scores can exceed the printed operating corner; unpoisoned
ones miss it. This is a dependent 20-customer pilot, not full reproduction,
validated calibration, or an all-configuration bound. All 20 consumed arrays,
metadata, paired features/IDs/labels, 675 flips, output hashes, and metrics
passed local checks; local comparison equals cluster bytes. Pre-freeze main
suite: 307 pass, 4 Ada fit tests skipped; pinned environment and compute node
each passed all 7 Ada tests. Separate setup job 398338 took 67 seconds after
login-node venv bootstrap was killed; no research fit there. Expected library
deprecation warning retained. Source correction: tree family is explicit in
III-B.2(b), only depth/details omitted. Old source hashes now check recorded
immutable commits when today's direct implementation files differ; old runs
and contracts were not overwritten. See results/adaboost_pilot_20260920.
Stop this pair; next proposed model is SVM with omitted gamma/coefficient/score
choices frozen first. No SVM or alternate Ada algorithm/depth/seed ran.
Journal/site draft updated locally, not deployed.

**Current matched-control step, completed:** SPLIT_RESAMPLING_CHECK.md and
code frozen at e6e0359; all 304 tests passed. User-authorized CPU job 398164
completed 0:0 in 1:48, within 15 minutes/4 CPUs/16 GiB, no GPU. A reuses old
RF scores; B preserves original rows but applies train-only ADASYN; C holds
out whole source days before train-only ADASYN. Exactly four new fits,
unchanged forest/seed/attacks/poisoning-customer selection. Common evaluation E
uses 445 original rows (386 attacks, 59 benign), 175 days, 20 customers,
selected by identities using seed 20260920/role 501. A/B/C AUC is
98.31606/91.84157/93.18082 at p00 and 90.98094/81.98823/82.97181 at p30.
The larger 1,288-original-row A/B comparison agrees in direction. Resampling
policy matters, but C does not collapse further and stays above daily-mean
AUC 67.62975. C's best DR at FA<=17.6% is 91.19171/69.94819; its AUC
falls 10.20901 points with poisoning. Training counts A/B/C 4464/4532/4538;
p30 flips 675/675/696 rows, same six customers. All 112 arrays, ancestry,
scaling/labels, common evaluation, C day exclusion, model reloads, and score
metrics passed cluster/local audits. Summary SHA256 fba6d6cd...b5ca866.
See results/split_control_20260920. No CI, unseen-customer claim, or full-data
reproduction. Preserve all original/control scientific files and outputs.
Stop this diagnostic; proposed next question is a specified AdaBoost pair,
not repeated seeds. No next model was launched. Journal/site draft updated
locally, not deployed. Panther main checkout and Paper 1 remain unchanged.

**Current first-baseline step:** user authorized the next steps after verified
preparation. FIRST_BASELINE.md freezes the 100-tree scikit-learn 1.9.0 forest,
defaults/seed, paired prepared p00/p30 hashes, ordinary accuracy, simple
controls, per-attack/benign-stratum results, all-cutoff diagnostics, and
save/reload agreement. One 15-minute, four-CPU, 16-GiB job is planned for two
fits on the existing 20-customer/28-day pilot. No data regeneration or sweep.
The direct models.py/run_experiment.py/analyze_results.py passed the full
296-test suite and were frozen at 48e979a. CPU job 397217 completed 0:0 in
16 seconds. The two forest fits took 0.649/0.566 seconds. DR/FA/AUC are
92.31/3.02/98.55 at p00 and 61.18/0.44/94.36 at p30. At FA<=17.6%, the
best saved-score DR is 97.29/92.04%; at FA<=33.3%, 99.19/95.02%. The
default DR decline is not loss of all useful ranking. These test-label-chosen
cutoffs are diagnostics, not validated calibration. Both fits pass exact
save/reload; hashes and metrics passed, and local/cluster comparison files
match byte-for-byte. See results/rf_pilot_20260920 and the new journal entry.
Original benign FA differs from synthetic FA (6.56 vs 2.33 at p00), and the
pre-split parent dependence remains. A matched split/resampling check is the
next useful diagnostic before scaling. No automatic extra seed or family sweep
ran. This useful baseline result must be reported plainly; it is not full
Table III reproduction or a population confidence statement.

**Paper 3 preparation implementation:** user approved steps 1 and 2. Fresh
official ISSDA metadata, six ZIP identities, and CRCs pass; the allocation
CSV independently matches 6,445 workbook rows, including 4,225 residential
IDs. Author's 3,000-ID selection remains unknown. Jokar's thesis Chapter 5,
p.117, supplies attack factor ranges and duration context; target attack
numbering and half-hour coordinates control. PREPARATION.md declares all
initial choices. download_data.py and prepare_data.py under the Paper 3
reproduction directory pass 13 fixture tests. ADASYN values come from the
stock library, with recorded neighbors and verified synthetic ancestry.
Novelty contamination plus all-M testing has measured-identity overlap by
construction, explicitly retained and reported. Panther login succeeded;
matching packages/raw sources are present. Frozen commit 30ce6c4 passed 284
tests and completed as CPU job 397206, exit 0:0, in 2:16 (15-minute cap,
4 CPUs, 16 GiB, no GPU). The wrapper passed all 13 fixtures, then scanned all
157,992,996 readings and retained 560 profiles (20 x first 28 complete days).
Eight cases completed and all 224 saved arrays passed an independent artifact
audit. Generalized two-class nominal 30% flips 675/4464 rows (15.12%); the
customer-specific denominator flips 67/224 (29.91%). Novelty contamination
shares 108 attack identities with testing in the generalized case, five in the
single-customer case. Synthetic-parent split crossings are also recorded.
These are preparation observations under declared choices, not performance
results or identified author behavior. See results/preparation_20260920.
No detector training/scoring or full-population preparation occurred. The
Panther code ran in an isolated checkout; Paper 1's checkout is unchanged.
Never persist authentication secrets in this record or repository.

**Run-selection rule, September 20:** a gross mismatch must trigger diagnosis,
not an automatic three-seed retry, larger grid, or longer fit. Inspect shared
setup, saved scores, metrics, actual updates, and controls first. Every extra
experiment must name its question, competing predictions, decision-changing
outcome, maximum cost, and stop rule. Repetitions remain appropriate for a
credible instability explanation or a predeclared uncertainty/statistical
question. A large gap does not itself prove low seed variability. All reported
models remain in scope; this rule limits uninformative retries. It is recorded
in the active plan and the canonical journal, which feeds the website draft.

**Journal requirement, September 20:** the user wants the actual research
route recorded internally and on the site: what we thought, why we chose a
check, what happened, and what changed. Canonical editable source:
`studies/takiddin-2021-robust-poisoning/RESEARCH_LOG.md`; render with
`scripts/render_robust_journal.py` (or `--check`). It preserves the four source
issues including the duplicated log(p) loss, explicitly marks unrun tests,
and includes a retrospective note on the earlier arithmetic audit. A local
website journal and homepage links are drafted but not deployed. Tone:
introductory-statistics undergraduate, connected prose and evidence, minimal
analogy; follow `docs/WEBSITE-JOURNAL-BRIEF.md`. The live homepage, current
report, expanded bound, earlier notes, water study, water PDF route, and
GitHub evidence link were visited in the browser. No matching public code
was found in the exact-title/DOI GitHub searches, author listing, and Zenodo
DOI search. IEEE's verification barrier prevented supplement inspection.
Preserve `CODE_AVAILABILITY.md`; absence from this search says nothing about
intent. No model experiment ran. Update the journal with every material step;
append corrections instead of silently overwriting the reasoning history.

**Current Paper 3 direction, September 20:** restart the reasoning from the
paper and reproduce every baseline and proposed model using standard libraries
and the reported settings. All seven baselines, both ensembles, all four
poison levels, and generalized/customer-specific tables are in scope. Plan:
`docs/plans/2026-09-20-robust-all-model-reproduction.md`. This turn is reading
and planning; no new experiment ran. The first priority is verified ISET data,
explicit poisoning/preparation, and faithful ordinary implementations. Fresh
visual reading verified Equation (1)'s duplicated log(p) term and the
sequential network's classification-only stated objective. Standard BCE is a
plausible correction; pretraining and reconstruction-loss alternatives need
explicit treatment. Table IV averages customers, which matters for nonlinear
metric identities. Earlier findings are preserved and must be reassessed,
not inherited as conclusions. Skip speculative numerical-pattern or
fabricated-data investigations. Prefer one V100-16GB when available, benchmark
actual throughput, and record both 50-epoch completion and performance at the
paper's reported time under a finite prospective budget. Panther responded to
SSH, but noninteractive authentication failed; no current GPU availability
was checked. Paper 1's separate task is outside this plan.

**Historical September 2 result; Paper 3 source-only:** the user selected Takiddin et al.,
“Robust Electricity Theft Detection Against Data Poisoning Attacks in Smart
Grids” (IEEE TSG 2021, DOI `10.1109/TSG.2020.3047864`) as the next independent
audit. The ten-page PDF SHA-256 is `03a372fb...3d4a4ff`; all pages, Tables I–V,
Figures 1–3, and Algorithm 1 were visually inspected. Historical plan:
`docs/plans/2026-09-02-robust-poisoning-source-audit.md`. The source
specification, causal map, and all 476 Table III–V metric cells were frozen
before the static arithmetic check. Results: SP 68/68 pass, F1 67/68, balanced
ACC 68/68, balanced PR 10/68, and the any-prevalence screen 65/68. Table V's
sequential-ensemble rows at 0%, 10%, and 30% poisoning cannot reconcile
DR/FA/PR with DR/FA/ACC under any one prevalence within one-decimal rounding.
This is a source-level internal inconsistency; it is not trained
non-reproduction, a mechanism result, or an intent claim. Preserve
`SOURCE_AUDIT_FINDING.md`, `EXPLANATION_REGISTER.md`, the corrected result hash
`a9113f6b...c52da56`, and the first reporting-incomplete attempt. The approved
source phase completed and stopped for discussion. No data preparation,
implementation, training, scoring, or cluster submission occurred in that
phase. The current September 20 direction appears above.
The paper explicitly reports 50 epochs, batch 100, NVIDIA RTX 2070, roughly
one hour for shallow models, 1.5–3 hours for deep models, three hours for
ensemble averaging, four hours for the sequential ensemble, and about two
seconds per online decision. Preserve these source statements. The September
20 plan governs later hardware and budget choices. Paper 1 transfers no verdict.

**Newest direction; paper time is the budget:** the user explicitly rejected
faster or additional GPUs as a rescue. Table IV reports 183 minutes for
full-ISET LSTM-SAE training while omitting hardware, GPU count, epochs, batch,
versions, stopping, timing boundary, and repetitions. Contemporaneous primary
Texas A&M records place K80/V100 devices in the plausible available range but
do not identify the paper's actual device. Freeze exactly one V100-16GB, one
seed, batch 32, exact full data/model, and 10,980 seconds of fitting followed
by full scoring. No H200/A100, multi-GPU, retry, or longer fit. See
`PAPER_TIME_BUDGET_CONTRACT.md` and
`docs/plans/2026-09-01-paper-time-budget.md`.

**Newest completed result; publication authorized:** the frozen paper-time runner
at `46f0ddd` completed as Panther job `385632` on one V100-16GB. The exact
183-minute fit completed 1.268 epoch-equivalents. Printed-cutoff DR/FA/AUC are
16.62%/31.91%/40.30%; at `FA<=13%`, maximum DR is 7.00% in the paper direction
and 23.02% reversed, versus the reported 85%. No cutoff recovers the corner.
All 8,884,989 finite scores, hashes, counts, and AUCs passed independent audit.
One V100 epoch took 143.883 minutes; ten project to 23.98 hours, 7.86 times the
paper budget. Loss still declined, so preserve bounded time/completion and
“ordinary additional training is implausible,” not unlimited-time
impossibility. See `PAPER_TIME_BUDGET_FINDING.md`. The result discussion is
complete. The current HTML report, homepage, and README now explain the full
row, exhaustive cutoff result, runtime evidence, alternatives, and limits; the
older PDF is explicitly labeled as ending on 1 September. Do not submit a
retry or another experiment. Stop after deployment to discuss whether the next
question should be mechanism, breadth-first remaining-model gates, or a
longer-time attainability test. Publication verification passes all 265 tests
(140 study and 125 root), strict data identity, source/link checks, and
whitespace checks.

**Superseded hardware-promotion result:** after discussing what would make a
result highly implausible versus fabricated, the user authorized the proposed
prospective LSTM-SAE gate, cost check, and one full anchor if affordable. The
preserved A16 pilot projects 42.7902 hours for ten epochs and 417.1425 hours for
100, so it fails the frozen 72-hour gate. The approved single-H200 pilot was
submitted once as job `385602` at frozen commit `93ecd0d`, but Slurm held it
because the account's only QOS (`gpulimit`) is not permitted on the H200
partition. It was canceled while pending with zero GPU time. No model, epoch,
score, or artifact was produced. The required projection was therefore not
observed, so the full anchor was not promoted. This is operational `X`, not
paper evidence. No other model, mechanism wave, table, seed, or publication is
included. The later paper-time direction closes hardware authorization as a
next step and replaces it with a fixed, historically plausible time envelope.
See
`LSTM_SAE_ANCHOR_PROMOTION.md` and
`H200_COST_FINDING.md` and
`docs/plans/2026-09-01-lstm-sae-anchor-promotion.md`. The evidentiary language
boundary is saved in
`docs/decisions/2026-09-01-implausibility-and-fabrication-boundary.md`.
The bounded instrument passed all 256 repository tests (140 study
and 116 root), strict data verification, compilation, shell syntax, and
whitespace checks. Historical `run_experiment.py` and
`remaining_models.py` hashes remain unchanged.

**Newest direction, 2026-09-01:** update and publish the full site with the
completed Sigmoid evidence first, then finish the paper audit. Active plan:
`docs/plans/2026-09-01-full-site-and-paper-completion.md`. Publication includes
README, all current ATK site entry points, and a newly current rendered PDF.
No science runs during publication. Afterward freeze exact remaining Table-III
proposed-model, mechanism, Tables IV/V, and finite Table-II contracts and show
their costs/ambiguities at the recorded checkpoint before cluster submission.
Historical breadth remains exploratory/quarantined. Publication commit
`b1973d9` is live through successful Pages run `33448349593`; all 11 deployed
files match local bytes. The bundle passed 235 deterministic tests, data
identity selected the verified ScienceDB branch, and all five PDF pages were
visually inspected. Phase A and the Phase-B checkpoint are complete; Phase C's
feasibility wave is complete and stopped at the discussion checkpoint below.

**Phase-B checkpoint, 2026-09-01:** the direct-paper reconstruction is saved in
`studies/atk-2022-deep-autoencoder/REMAINING_PAPER_CONTRACT.md`. The paper fixes
the remaining Table-I widths/optimizer/dropout/activations and printed cutoffs,
but not an executable recurrent decoder schedule, VAE probability, or AEA
loop. Proposed primary choices are first-step latent then zeros/top-state-only
for LSTM-SAE/VAE; Equation-(10) sum-squared plus KL with ten-draw probability
kernels for VAE; and autoregressive, concatenated additive attention for AEA.
Batch remains 32. First wave: four two-hour cluster feasibility jobs, eight
GPU-hours maximum; each passing full anchor has a 72-hour cap. The user accepted
all five checkpoint choices on 2026-09-01. The pilot implementation is isolated
in `reproduction/remaining_models.py`; do not edit the historical
`reproduction/models.py`, whose SHA-256 is bound into the Sigmoid-fit evidence.
The exact pilot uses the first 32,768 entries of `table_iv_order.npy` and 12,119
test rows: 1,024 source days with all seven benign/attack identities plus 4,951
synthetic benign rows. Each job must use the pilot-only two-hour `gpu-short`
wrapper, batch 32, two epochs, exact save/reload, two safe scoring partitions,
and the recorded 72-hour projection/memory gates. The VAE pilot scores MC10 on
all 12,119 rows and exercises MC1/10/100 on 256 interface rows; the primary
full anchor must still perform the contract's 12,119-row MC stability check.
All 243 deterministic tests pass (140 study, 103 root), the historical model
source hash remains exact, and strict data verification selects the complete
ScienceDB semantic-equivalence branch. Frozen/pushed commit `0ca6cc4` was
synchronized to Panther. Jobs `385544`--`385547` then all stopped before model
construction: cache `metadata.json` omits the expected checksum entry for
`table_iv_order.npy`. Its actual hash exactly matches the eligible anchor's
committed record, so this is a manifest-source bug, not corrupt data. Zero
models/updates/scores; 278 GPU-seconds total. Preserve
`REMAINING_PILOT_FINDING.md` and
`results/remaining_pilot_preflight_20260901.json`. Discuss the narrow
anchor-manifest fallback before any code repair or retry. The user approved
that exact repair on 2026-09-01. It must verify the committed anchor-result hash
and matching metadata hash, record every manifest source, and create a new
tested commit and immutable attempts; every other field remains frozen. The
narrow repair passes 245 deterministic tests (140 study, 105 root), and strict
data verification selects the complete ScienceDB semantic-equivalence branch.
New jobs `385552`--`385555` then completed: FC-VAE passed every feasibility
gate (44.81-hour conservative full-anchor projection); LSTM-SAE failed the
`1e-6` all-score batch-agreement gate at `1.838e-5`; LSTM-VAE failed only its
auxiliary MSE-plus-KL score at `1.967e-6` while its primary differed by
`1.178e-7`; LSTM-AEA passed all but runtime, projecting to 1,879.93 hours for
100 full-data epochs. All arrays are finite and transferred artifact hashes
match. Total exposure was 0.8 GPU-hours. This is operational `X`; stop for
discussion before promotion, tolerance changes, AEA changes, or publication.
The complete resolver also exposed missing metadata entries for
`test_attack_id.npy` and `test_source_row.npy`; the same exact anchor supplied
all three absent hashes and the actual bytes matched.

**Newest result; discussion required:** the approved score-only recovery is
complete. Jobs `385583` (LSTM-SAE) and `385584` (LSTM-VAE) ran commit
`43abc09`, completed `0:0`, and used no training. Across score batches
256/128/64/32, printed-cutoff labels and metrics were identical. SAE's ROC
summary was identical. VAE AUC moved at most `0.0000217922` percentage points;
best balanced accuracy and FA-capped DR/FA stayed identical. An exact
batch-256 ROC cutoff transferred to another batch can change one boundary row,
so preserve “near decision-invariance,” not absolute invariance. The original
`1e-6` all-score gate remains failed and unchanged. All transferred hashes,
array shapes, and finite checks passed; 227 GPU-seconds total. Post-result
verification passes all 252 tests (140 study and 112 root) and the strict data
gate. See
`REMAINING_SCORE_RECOVERY_FINDING.md` and the completed bounded plan. Stop
before gate replacement, promotion, retraining, FC-VAE, AEA, or publication.

**Earlier completed Sigmoid work:** the user requested testing Sigmoid plus another
cutoff. The bounded paired fit is complete and stopped for discussion; see
`SIGMOID_FIT_FINDING.md` / `docs/plans/2026-08-31-small-sigmoid-fit.md`.
Frozen `cc9af5e`, 230 pre-run tests, CPU job `385198` completed `0:0`.
Pilot 9.2163 s; measured estimate 62.2256 s (92.0258 s conservative) promoted
the sole small pair. Actual small analysis 24.8064 s, total allocation 3:52
including slow pilot startup. Both heads completed ten epochs / 640 updates
with identical initial weights. On 12,119 sampled held-out rows, max DR at
FA<=15% is Softmax 8.64258%, Sigmoid 9.74935%; reversing gives 25.52083% /
25.39063%, all below 81%. Rounded targets also fail. This is an exact
all-cutoff exclusion for these fixed scores/sample, NOT all Sigmoid weights.
Sigmoid calibration MSE improved 1.61028→1.33863; best epoch is 10, the last
tested. No long-run plateau, zero-useful-work, or seed-probability claim is
earned. All hashes, finite checks, and weight updates passed. Preserve all
outcomes. The earlier stop for discussion was satisfied; the user approved a
full public update on 1 September before the remaining-paper program. No new
science run has started.

**Previous completed work:** the discussed source-assumption findings are live
at `dc37bbe`; Pages run `33419150100` succeeded and nine public files match
local bytes. The subsequent quick Sigmoid investigation is complete and LOCAL
ONLY, awaiting discussion. See `SIGMOID_SANITY_FINDING.md` and
`docs/plans/2026-08-31-source-findings-and-sigmoid-sanity.md`.
Frozen code `9d6c31b`, CPU job `385137`, pilot 4.97 s, full 39.70 s, allocation
2:17, completed `0:0`. No training, head swap, rescaling, or regeneration.
On all 8,884,989 prepared rows, Sigmoid + printed cutoff 0.58 still fails:
minimum FA is 29.66640%, versus 15%. With a changed cutoff, however, its bound
allows up to 85.32587% DR at FA<=15%, so the pair is NOT excluded. Reversed
scoring also remains open (upper DR 93.76498%). This is a bound, not an achieved
model result. Original-row high-error DR ceiling remains 59.98410%; adding
synthetic benign rows changes the FA population, so do not generalize the
original-row exclusion. The best cutoff-based balanced accuracy among clipping
and constant-half controls in both directions is about 58.23%; neither
is a learned Sigmoid model. Preserve these openings as plainly as failures.
Stop for discussion; no new public edit/push, fitted comparison, or experiment
is automatically approved. A head swap in Softmax-trained weights alone is
not a test of Sigmoid learning.

**Earlier checkpoint, now satisfied:** the user required discussion of measured
results before updating the public account. That discussion occurred, the
Sigmoid range and fitted findings were added, and the 1 September instruction
now governs publication. The earlier source review was limited to the cheap
assumption checks in `SOURCE_ASSUMPTION_CHECK.md`:
job `385119` exited `0:0`; full analysis 56.76 seconds after an 8.20-second
pilot. Code/contract frozen in local `b76cb02`; no push or public edit occurred.
See `SOURCE_ASSUMPTION_FINDING.md`. Softmax + high-error MSE are explicit;
normalization statistics are omitted. At the printed 0.58 cutoff, upper DR is
29.58% current, 29.81% joint scalar, 33.96% separate-class feature scaling
(rounded upward), versus 81%. These attack-only bounds survive benign-only
resampling on unchanged attacks. All-cutoff alternative limits use original
rows only, NOT regenerated ADASYN. Sigmoid-range control raises the original
ACC ceiling to 80.84% but still fails DR/FA jointly in the printed direction;
with reversal its relaxation no longer excludes the DR>=81%, FA<=15% pair.
Those changes are not the stated FC-SAE and prove no trainable-model match.
No new trained numerical or mechanism experiment was performed. Await discussion.

**Current state — diagnostic round complete, experiments stopped:** the user
approved the bounded post-anchor sequence on 2026-08-31. Jobs `385090` and
`385091` both completed `0:0`; no training or original artifact changed.
The full analysis took 112.92 seconds, the adaptive 70,000-row control 18.03.
`POST_ANCHOR_FINDING.md` is the current interpretation; preserve the two frozen
contracts, code revisions `1175e8d`/`26a42db`, and raw diagnostic summaries.
No further experiment is authorized by completion of this round.
The follow-up is public: report commit `97c9236`, successful Pages deployment
`33407618030`, all nine checked public pages/assets byte-identical to local.
All 206 local deterministic tests passed, including 14 static report checks.

**What the new evidence says:** on the frozen prepared inputs, an optimistic
label-aware Softmax/MSE relaxation bounds ACC below 50.93%, AUC at 45.11%,
and DR at 9.25% when FA <=15%, versus 83/81/81% reported. This covers every
weight/seed under those assumptions, not other preparation/output/score
choices. It is an outward-padded float64 evaluation of an analytic bound, not
certified interval arithmetic or extrapolation over seeds.
Original-row trained-minus-zero ACC is +0.89081 points [0.80454, 0.98117]
under a fixed-model, customer-cluster bootstrap. Some per-attack gains exceed
one point. The adaptive within-energy control gives trained/projection/
uniform/energy AUC 65.49/62.18/55.02/49.74%; no CI for that statistic.
Thus “no useful work” is not established. Score comparisons do not identify
training causally or test the paper's recurrence/attention explanation.

**Public-writing direction, 2026-08-31:** the user wants the README and site to
read like a clear research report: paper → implementation → result → possible
explanations → tests → bounded conclusion. No workflow jargon in place of
explanation. A separate current reproduction page is added; both older paper
pages remain. Obsolete charter commands and duplicate agent rules are removed,
not the scientific history. The user explicitly approved GitHub/GitHub Pages
publication on 2026-08-31. The rewrite and supporting evidence were pushed;
Pages run `33400529269` succeeded and all public pages/assets match the reviewed
files. The README links to the rendered reports. At that first publication,
all 187 local tests passed,
including eight static report checks. The subsequent limited diagnostic
authorization is stated above; no full training or broad search is authorized.
See the 2026-08-31 editorial plan and the completed diagnostic round above.

**Next scientific emphasis:** discuss which source-supported assumptions could
change the bound, beginning with a source/semantic map, not another seed of the
same setup. Any further experiment needs a recorded question and approval.
Rescaling an accuracy axis does not make a target less probable; fixed-model
confidence intervals do not bound new seeds/configurations, and years-of-search
estimates are conditional. The completed analytic bound avoids those shortcuts.

**Governing plan and historical progression (current boundary above):**
[`docs/plans/2026-08-23-clean-reader-reproduction-rebase.md`](plans/2026-08-23-clean-reader-reproduction-rebase.md)
supersedes the prior model-family/one-factor execution sequence. The user
approved the plan on 2026-08-23. Phase 0 preservation/reconciliation is
complete; Phase 1 paper-only orientation is also complete and recorded in
`studies/atk-2022-deep-autoencoder/CLEAN_READER_ORIENTATION.md`. The user
authorized Phase 2's bounded disposable sandbox on 2026-08-24. Only the
pre-recorded toy/synthetic `X` wave in `DISCOVERY_SANDBOX.md` may execute;
named-data and formal work remain closed. The wave's frozen contract,
standalone script, and Slurm wrapper are in `83dab57`; 140 study tests and 33
root tests pass. Panther job `381540` completed the one authorized wave with
exit `0:0`; Phase 2 is complete and its results remain exploratory `X` only.
Phase 3 is also complete: all 12 PDF pages were re-inspected and
`studies/atk-2022-deep-autoencoder/CLEAN_READER_SPECIFICATION.md` freezes the
literal failures and candidate `CR-ISET-FCSAE-01` completion. The user approved
Checkpoint 1 on 2026-08-24 and directed the six-step sequence to proceed while
requiring every plausible explanation to be preserved and tested
systematically. Phase 4 is complete: the five-file trace, quarantine, minimal
corrections, fail-closed contract, and 179 deterministic tests are recorded in
`studies/atk-2022-deep-autoencoder/CLEAN_READER_FIDELITY.md`. The former Phase-5
exact-serialization block remains preserved. Official metadata identifies
the `.tab` as Dataverse's ingest of an original XLSX; the allocation mapping is
not missing. A public ScienceDB CSV and public GitHub workbook agree across all
6,445 mappings, and every residential reading ID is covered. The exact archive
bytes are already available. On 2026-08-30 the user explicitly approved
`sciencedb-csv-semantic-equivalence-v1` as the visible allocation `I` branch
for the otherwise unchanged single anchor. Phase 5 is operationally complete;
the sole attempt ran from eligible code commit
`a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`. Phase 6 is complete: frozen audit
and 65 corrected supplemental checks passed; the initial Checkpoint-2 pause
was later opened only for the now-completed diagnostic round above.
The 179 deterministic tests and
strict local seven-file verifier passed before submission. Panther job
`384390` completed with exit `0:0` at 22:48:39 Qatar time on 2026-08-30 after
9:14:27. The thread heartbeat `monitor-clean-reader-anchor` is paused at the
user's request; checks remain manual. No second submission or experimental
change is authorized. The full frozen run contract is
`studies/atk-2022-deep-autoencoder/CLEAN_READER_ANCHOR_PRERUN.md`, tied to code
commit `a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`. Checkpoint 2 remains binding.
Initial contiguous beginning/middle/end array samples were finite and varied;
do not describe sampled checks as an exhaustive NaN/Inf scan. The completed
preparation records 1,500,523 training and 8,884,989 post-ADASYN test rows.
Actual ADASYN time was 1:04:51, preparation about 1:09:43, fitting 8:02:46
(28 epochs, best epoch 23, Tesla P100), and scoring 11.12 s. The former
07:00--08:00 Aug-31 ETA was wrong: the neighbor benchmark used a larger
historical population, and old training timing did not transfer to this
hardware/contract. The result is at
`clean-reader-v1-results-semantic-allocation/runs/table_3/fc_sae/seed_20260824_2f483335536c/result.json`
under the remote reproduction-derived directory. The independent audit now
regenerates all metrics and confusion counts with zero discrepancy. DR/FA/ACC/
AUC/F1 are 25.48/45.13/40.18/39.40/30.09% versus 81/15/83/81/81% reported.
This is one frozen `P+I/N` non-reproduction, not a paper-wide or confirmatory
finding. See `studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md`.
Full scans of 31 arrays found no NaN/Inf; 2,816 training and 1,409 test customers
are disjoint; 256 CPU fresh-load scores agree within 1.20e-7. The checker-v1
last-chunk boundary bug and failed report are preserved; v2 clamps the slice
and passes, without changing any reproduction artifact.
All-threshold balanced ACC is at most 50.00072% in the paper direction and
60.21% reversed for these fixed scores. Trained/zero-score Pearson correlation
is 0.999253, with only +1.18 ACC points over the zero rule; the Softmax-domain
floor has ACC 40.20 versus 40.18 trained. These motivate geometry/scoring
questions but establish neither causal mechanism failure nor a method-family
attainability envelope. No further experiment is approved.
The preceding fixed-score interpretation is the pre-diagnostic state; the
conditional bound and useful-information measurements above supersede it.
The user reconfirmed that sandbox tests should remain quick: Phase 2 took
60.06 s of computation (2:25 total job); this 9:14:27 run was the later full-data
numerical reproduction, not sandbox breadth. Explain that boundary explicitly.
Older execution-order notes below are historical unless the new plan explicitly
promotes them.

**Checkpoint-1 focus:** the candidate first anchor is Table-III ISET FC-SAE,
seed 20260824. It uses exact-byte official ISSDA V1 consumption archives plus
the approved semantic-equivalence allocation CSV; duration-first in-day Attack
3, joint pre-split feature scaling,
customer-disjoint B1/B2 with attacks restricted to B2, test-set ADASYN,
`48-400-300-200-100-100-200-300-400-48` sigmoid/Softmax FC-SAE, dropout after
all hidden layers, an explicit training-loss convergence completion, printed
threshold 0.58, and one attempt only. Controls and VAE/LSTM/attention repairs
remain separate and deferred.

**User emphasis after Phase 2:** the most interesting sandbox observation is
that the LSTM performed worse than the dense AE on a deliberately temporal toy
task. Treat this as a rational increase in concern about the paper's mechanism
story, not a paper-level result: the LSTM underfit, and a dense network over 48
ordered coordinates can also model temporal relationships. The next formal
mechanism program, if promoted, must separate fitting success, temporal
structure, inductive bias, and paired advantage. The VAE direction conflict may
be a typo; preserve both directions and test them rather than overinterpreting
the prose.

**Explanation memory:** use
`studies/atk-2022-deep-autoencoder/EXPLANATION_REGISTER.md` as the live
hypothesis ledger. It contains E1–E6 from the temporal discussion and E7–E15
for geometry, scoring, preprocessing, data, reproduction error, witness
weakness, variation/selection, reporting error, and attainability. Add
plausible explanations with competing predictions; never silently delete one.

**Source-only findings now durable:** the 2026-08-23/24 discussion is recorded
in `docs/EVIDENCE-AND-LEARNINGS.md` under “Clean-reader source-only findings
before the discovery sandbox.” It preserves six separate statements: tidy
total ordering is an audit signal rather than an accusation; the reported
full-configuration comparisons do not identify the credited mechanisms; task
triviality is a sandbox hypothesis; multiple written operations are
non-executable or underdetermined; standardized targets plus Softmax/sigmoid
outputs imply a genuine reconstruction-error floor but no direct metric bound;
and all three formal `N/M/A` verdicts remain open.

## Environment quirks

- Table-II seed-11 breadth completed on Panther on 2026-08-18. Literal SGCC is
  non-executable because 1,034 daily inputs are never mapped to the printed 48
  half-hour architecture. All eleven named rows ran on `last_48`; all five
  proposed rows plus feed-forward ran on `first_48` and `binned_mean_48`.
  Proposed AUC is 46.31--54.15 across the complete representation matrix versus
  83--93 reported. Feed-forward AUC is 95.31--96.91 and its best DR/FA gaps are
  only 0.80--1.94 points, so absent signal is not a sufficient explanation.
  Exact all-threshold complete-vector gaps for proposed runs are 33.18--50.53
  points. Pairwise proposed-model Spearman score correlation is at least 0.957
  within every representation. Results:
  `studies/atk-2022-deep-autoencoder/TABLE_II_BREADTH.md`.
- Static source arithmetic is stronger than before: because the paper calls
  its ADASYN test output balanced, precision must equal `DR/(DR+FA)`. Five
  Table-II rows and eight Table-III rows fail that identity even under generous
  ±0.5-point rounding. The earlier draft claim that balanced accuracy lower-
  bounds AUC was wrong and is explicitly invalidated; the correct one-point
  bound is `AUC >= TPR*(1-FPR)`, which the printed rows satisfy.
- Table-V common-model/common-benign breadth is complete for all proposed ISET
  models. Each reproduced FA is exactly constant over attacks 1--6 (FC-SAE
  58.22, LSTM-SAE 40.96, FC-VAE 32.62, LSTM-VAE 25.79, LSTM-AEA 58.22), while
  the paper varies FA by attack. This is a confusion-matrix identity, not a
  stochastic observation. Alternative retrain/resplit experiment identities
  remain open. Table-IV jobs 378182--378191 cover the ten missing half and
  three-quarter cells and were all running when this context was updated.
- Exact all-threshold audits for the five proposed ISET score vectors leave
  complete-row minimax gaps of 49.96, 48.91, 55.04, 58.48, and 60.11 points in
  model order. Threshold choice is therefore eliminated for these vectors.
  The compact runner initially omitted Table-IV target dictionaries for
  FC-VAE/LSTM-VAE/LSTM-AEA and non-FC Table-V targets; commit `fcd2d78` restores
  the complete transcription and a coverage test. The omission affected only
  post-score result writing, not training or saved scores.
- The first predeclared one-factor data branch is now executable in the compact
  five-file route: all 4,225 residential meters versus a deterministic seeded
  3,000. It has not run. Panther is temporarily unreachable because the saved
  QCRI VPN connection requires an interactive OTP; never substitute a local
  preparation or experiment.
- Compact batch-512 Paper-1 anchor job 373789 completed 2026-08-11 in 53:12 on
  one V100. `I-ADASYN-NONE-ISET-FC-SAE`, seed 11, reproduced
  DR/FA/ACC/AUC/F1 = 26.18/58.22/33.98/31.04/40.46% versus
  81/15/83/81/81% reported. It closely repeats the batch-32 sensitivity rather
  than rescuing it. Since printed ISET ADASYN adds benign rows only, it cannot
  change the observed malicious DR; even FA=0 would cap balanced ACC at 63.09%
  for this trained model. Score audit job 373800 found that an oracle threshold
  in the paper direction reaches only 50.00% ACC, reversed direction reaches
  66.26%, and the learned score ranking is 0.99946-correlated with the
  zero-reconstruction control. This is an eligible exploratory interpretation,
  not completed printed ADASYN or a paper-wide verdict.
- Unchanged Softmax repetition jobs 373803 and 373804 were cancelled before
  execution on 2026-08-11. The user corrected the sequence to breadth-first:
  one sound anchor already establishes a large gap, so first locate which
  one-factor data/model/evaluation choices cause divergence and cover the model
  families; repeated seeds belong after that map exists. Queue diagnosis also
  found three V100s free, but only ten CPU cores free on their node while each
  job requested sixteen; the delay was resource shape/priority, not exhaustion
  of every 16-GB GPU.
- Breadth-first job 373805 completed the seed-11
  `C-OUTPUT-LINEAR-ISET-FC-SAE` one-factor control in 1:05:43; audit job 373824
  completed in 24 seconds. Linear output produced DR/FA/ACC/AUC/F1 =
  12.32/30.78/40.77/28.14/21.61%. A paper-direction oracle reaches only
  50.04% ACC; reversing direction reaches 67.56%. Benign mean error 0.537 is
  above malicious 0.281. Correlation with zero reconstruction fell to 0.82089,
  so the model changed materially but still did not yield the claimed score
  behavior. Table-V FA is exactly 30.0696% for all attacks on the common
  all-benign population. Output activation alone is therefore not a sufficient
  explanation for the gap. This remains a one-seed corrected control, not a
  paper-level verdict. Future short Slurm wrappers request only the needed GPU
  type/count and leave CPU/memory shape unspecified.
- **Historical, superseded 2026-08-23:** continue model-family breadth and then
  one-factor data/evaluation interpretations. The completed artifacts remain
  preserved, but this is no longer the active next-action sequence.
- The 2026-08-11 reuse audit found no preserved ISET/Table-III benchmark
  attempt in committed results or Panther manifests. Historical benchmark
  completions are SGCC/Table II and are ineligible for the current ISET rows.
  Start benchmark breadth with the ISET Naive Bayes row; do not silently reuse
  the SGCC numbers.
- ISET Naive Bayes job 373833 completed on 2026-08-11 from commit
  `3f9b73e` in 1m36s. It is the explicitly labeled
  `I-SUPERVISED-ADASYN-NONE-ISET-NAIVE-BAYES` row: GaussianNB with default
  `var_smoothing=1e-9`, positive probability threshold 0.5, all original
  all-customer `B+M`, and an exact seed-11 2:1 random row split. It does not
  execute the paper's pre-split supervised ADASYN and cannot fill the printed
  cell. It reproduced DR/FA/ACC/F1/AUC =
  88.78/44.53/72.12/90.50/79.17% versus 73/18/77.5/73/70% reported.
- **Historical execution instruction (2026-08-12):** independent frozen breadth
  rows ran concurrently, one generic GPU per job, up to the established
  three-job limit. This does not authorize new jobs under the 2026-08-23 plan.
- ARIMA breadth job 373836 completed in 1m02s using pooled ARIMA(1,1,0)
  residual MSE on full B1/original B2+M. DR/FA/ACC/F1/AUC =
  21.48/57.20/32.14/34.46/24.72% versus 86/12/87/86/87% reported. This is a
  strong non-match for the named completion, not closure of the paper's omitted
  autoregressive order, fit unit, or score branches.
- One-class SVM breadth job 373837 completed in 1m04s using the named
  sigmoid/scale repair, `nu=0.5`, 12k train rows, and 30k test rows. It produced
  DR/FA/ACC/F1/AUC = 91.87/50.94/70.47/94.35/79.67% versus
  90/9/90.5/89.5/87% reported. The close DR comes with >50% FA, so the fixed
  operating point is a non-match. The caps make this diagnostic rather than a
  full-cell result.
- Score-audit jobs 373854/373855 analyzed the three completed benchmark score
  vectors. NB oracle ACC is 74.74%; its closest ROC point to reported DR/FA is
  71.82/23.00, a minimum 5.00-point max gap, so NB mainly exposes an omitted
  operating-point choice rather than a catastrophic ranking failure. Pooled
  ARIMA's paper-direction oracle ACC is only 50.00%; reversing its score reaches
  69.74%, and its closest paper-direction DR/FA point remains 56.56 points away.
  Capped one-class SVM oracle ACC is 73.86%; its closest DR/FA point remains
  18.31 points away. Do not use per-row p-values: meter/day and six attack
  siblings are correlated. Confirmation needs repeated seeds and meter-level
  clustered uncertainty.
- Supervised feed-forward breadth job 373838 completed in 1:29:56 from commit
  `6a5baeb`: printed five-by-500 ReLU/Adamax architecture, predeclared
  two-Softmax categorical head, no supervised ADASYN, seed 11, batch 512.
  Fixed-threshold DR/FA/ACC/F1/AUC =
  96.41/23.72/86.35/96.24/97.05% versus 90/11/89.5/89.5/88% reported. Audit job
  374255 shows that threshold 0.824 gives DR=91.83/FA=9.17 (1.83-point maximum
  gap from the paper pair) and the best balanced threshold reaches 91.66% ACC.
  Treat this as strong ranking plus an omitted supervised threshold procedure,
  not as a fundamental model failure.
- Multiclass SVM job 373840 completed in 2m27s with deterministic 30k/30k caps:
  fixed DR/FA/ACC/F1/AUC = 85.94/55.67/65.14/88.04/73.06% versus
  91/8/91.5/90.5/89%. Audit 374302 gives oracle ACC 71.14% and a minimum
  23.44-point joint DR/FA gap. FC-VAE job 373842 completed in 6m14s:
  DR/FA/ACC/F1/AUC = 11.51/32.62/39.45/20.32/30.13% versus
  88/11/88.5/88.5/85%. Audit 374303 gives paper-direction oracle ACC 50.00%,
  reversed ACC 66.70%, and 0.99957 correlation with zero reconstruction.
- Recurrent operational failures are preserved in
  `results/recurrent_breadth_operational_failures_20260811.json`. Supervised
  LSTM job 373839 completed training but OOMed scoring at batch 8192. LSTM-SAE,
  LSTM-VAE, and LSTM-AEA jobs 373841/373843/373844 failed before training
  because diagnostic inventory assumed one output tensor. Commit `c735dd9`
  changes no scientific model: it records lists of output shapes and uses
  recurrent score batch 512. Replacement jobs 374310--374313 are sequentially
  queued from that commit. Jobs 374306--374309 were cancelled while still
  pending because Panther initially targeted a stale deleted upstream branch;
  no stale-code job ran.
- Supervised-LSTM replacement 374310 completed in 6:54:51 from `c735dd9`, six
  epochs with epoch-1 weights restored. Every test score is exactly 1.0:
  DR/FA/ACC/F1/AUC = 100/100/50/92.32/50% versus 90.5/10/90/90/89% reported.
  Audit 374387 gives oracle ACC 50% in either direction and a 90-point minimum
  DR/FA gap. A second operational leak was then found: the proposed-model
  pre-training sanity probe still materialized 10,000 recurrent rows at once.
  Commit `4469a53` batches that unchanged diagnostic by `score_batch`; focused
  tests pass. Jobs 374388--374390 were safely rejected by the immutable-attempt
  guard because score batch 512 retained the same failed-attempt identity.
  Wrapper commit `f5b5623` accepts an explicit score batch. LSTM-SAE job 374391
  completed and audit 374433 shows paper-direction oracle ACC 50.004%, reversed
  ACC 64.38%, and a 47.11-point minimum DR/FA gap. LSTM-VAE job 374395 trained
  through epoch 23 (best epoch 18) but OOMed during scoring; no-gradient
  fresh-process recovery 374441 scored the preserved weights at the original
  recorded inference batch without retraining. Fixed DR/FA/ACC/AUC =
  10.02/25.79/42.11/29.83%; audit 378014 gives paper-direction oracle ACC
  50.002%, reversed ACC 66.93%, a 58.48-point minimum DR/FA gap, and 0.93379
  correlation with zero reconstruction. LSTM-AEA job 374396 completed 100
  epochs in 43:41:50. Fixed DR/FA/ACC/AUC = 25.43/58.22/33.60/29.93%; audit
  378015 gives paper-direction oracle ACC 50.002%, reversed ACC 66.52%, a
  60.11-point minimum DR/FA gap, and 0.97843 correlation with zero
  reconstruction. All eleven Table-III model-family breadth rows are now
  closed for the registered one-seed no-test-ADASYN completions; no Paper-1
  jobs are active.
- Renewed Algorithm-2/4 check found that the earlier compact LSTM-SAE used a
  repeated latent but omitted the printed encoder-to-decoder hidden/cell state
  transfer. The compact recurrent builders now use mirrored state transfer.
  The ongoing decoder input remains omitted by the source and the selected
  breadth completion is explicitly repeat-latent. VAE reconstruction
  probability and Algorithm-5 attention repairs are likewise named in each
  configuration rather than silently called literal.
- The compact model-family breadth map is
  `studies/atk-2022-deep-autoencoder/TABLE_III_BREADTH.md`. None of the eleven
  fixed operating points reproduces its complete printed pattern. Supervised
  feed-forward is nevertheless a strong positive control; it and Naive Bayes
  expose an omitted operating-point rule. The other nine score vectors remain
  materially far from their reported DR/FA corner in the registered direction.
  Next: one-factor population, split,
  scaling, validation-threshold, and Attack-3 interpretations before seeds.

- Host is Apple M1 Max/macOS; public setup uses root `.venv` while the pre-publication workspace still has a legacy `replication/.venv`.
- Paper 1 neural runs use Keras 3 with the Torch backend and available Apple MPS; the paper does not state its backend, software versions, hardware, epochs, or batch size.
- Official CER/ISET consumption archives are restricted by ISSDA. Exact
  ScienceDB copies are local under `data/raw/cer-sciencedb/`; all six pass the
  official size/MD5 and ZIP gates. Its 6,445-row allocation CSV is not the
  official `.tab` binary but has zero normalized semantic mismatches against a
  second public allocation workbook and complete coverage of all residential
  reading IDs. Decision `2026-07-21-cer-sciencedb-semantic-allocation.md`
  admits it for the named exploratory branch only.
- Built-in macOS `unzip` failed on the multipart SGCC archive; 7-Zip 26.02 verified and extracted it successfully.
- In zsh, lowercase `path` is tied to `PATH`; never use it as a loop variable because system commands disappear for that shell.
- Cluster access is SSH-key-only and normally requires the institution's VPN
  off-site. Host, user, and project paths are deliberately not recorded in this
  public repository; keep them in local configuration.
- 2026-08-24 Phase-2 gate: the cluster was reachable and the `panther` host key
  matched already trusted aliases, but `ssh-add -l` reported no identities and
  batch authentication failed. The password prompt was cancelled; no remote
  command or job ran at that point. The user then clarified that Panther uses
  interactive password authentication. Interactive login and transfer
  succeeded; the credential was not persisted. Do not record it in project
  files or repeat it in summaries.
- Phase-2 job `381540` completed in 2:25 with exit `0:0` on one GPU. Raw JSON
  SHA-256 is
  `cef6e4d18ac765dcd5ba02b79c5deb51eace393c2670bee05e1fd54e577f2da8`.
  The sandbox found simple toy shortcuts for attacks 1--5, no confirming
  recurrent temporal witness, positive population-dependent decoder-domain
  floors, and low-probability VAE anomaly direction. These are `X`, not `N/M/A`.
- User execution policy (2026-07-24, widened 2026-07-24): run **every**
  experiment's preparation, training, and scoring on the cluster's compute nodes,
  never on the local Mac. This is not scoped to Paper 1 — it covers all
  studies and any private workstream. Local work is limited to code,
  documentation, lightweight inspection, and transfer/monitoring. Do not infer
  permission for a local fallback when the compute cluster is temporarily unreachable.
  Results produced locally are ineligible as experimental evidence and must be
  re-run on the compute cluster.
- The initial one-T4 batch-512 LSTM-SAE/LSTM-VAE OOMs and the cancelled V100
  attempt remain resource evidence. A one-T4 batch-32 run is a separately
  declared sensitivity, never a substitute for the primary batch-512 result.
- The primary-batch LSTM-AEA attention call receives two local
  `[128, 1034, 200]` tensors and attempts a 101.96-GiB allocation per rank;
  smaller unspecified batches remain an ambiguity branch rather than a silent
  replacement for the primary batch.
- The production neural runner is one Python program plus a short `sbatch`
  wrapper. Four-rank DDP preserves global sample-mean gradients and records its
  shuffle/random-stream choices; FC-SAE, LSTM-SAE, LSTM-VAE, and supervised
  feed-forward have now completed through that path.
- A Torch-native supervised probe's BCE assertion was an invalid diagnostic:
  the compiled-Keras loss/Adam rerun completed with synchronized finite state
  and identical final parameters across all four ranks.
- Cluster job 348195 completed and verified 12 classical Table II attempts.
  Mean DR was 7.97% NB, 2.10% ARIMA, 61.78% one-class SVM, and 53.51%
  multiclass SVM versus reported 75%, 88%, 91%, and 92%; all registered
  complete metric patterns are `NOT_CLOSE_MATCH` in this exploratory branch.
- The cluster's observed NVIDIA driver `570.133.07` cannot run the CUDA 13 package
  selected by the earlier floating Torch constraint. The cluster is pinned to the
  official PyTorch 2.7.1 CUDA 12.6 build and jobs fail on invisible CUDA.
- Even with identical SGCC/config/package versions and seeds, ADASYN produced
  77,708--77,712 supervised rows as OMP/MKL thread settings changed. The
  four-GPU cluster branch fixes both to 2 and records runtime cardinalities;
  this tiny variation is an ambiguity, not an explanation for large metric gaps.

- Never edit `config/exploratory_reproduction.toml` mid-branch: its byte hash
  is `contract_sha256` in every run fingerprint, so any edit invalidates
  resume-skip for all SGCC Table II attempts. Freeze ISET-phase additions
  separately in `config/exploratory_iset.toml`.
- The 2026-07-21 blanket fidelity audit is **INVALIDATED**. A fresh source-first
  audit on 2026-07-23 found concrete counterexamples: FC-SAE has seven rather
  than the paper-described eight hidden transformations; FC-VAE does not
  instantiate the printed four-plus-four hidden layout; ISET supervised data
  uses heldout-only attacks rather than malicious data for all customers; and
  the stated VAE reconstruction-probability detector is not implemented.
  Existing results are retained but gated by
  `PAPER_TO_CODE_TRACEABILITY.md`.
- Exact ISET preparation completed 2026-07-22 using all 4,225 residential
  meters. Cache SHA-256 is
  `ab88f180feafb7351ef4530cba2e48a3cbc180af268f8b68016aefc50b98a987`;
  Table IV subsets equal 30.603M/45.905M/61.206M scalar readings, strongly
  supporting the interpretation of the paper's 30M/45M/60M labels.
- LSTM-SAE seed 11 completed on four 16-GB V100s with 2:24:26 Slurm
  elapsed (2:24:03 pipeline; 2:20:38 fit): DR 6.78%, FA 2.22%, AUC
  51.89%. Its score is effectively
  input energy under zero reconstruction (correlation
  0.999999999999996), and even an oracle test threshold gives only 55.52%
  balanced accuracy. This is strong exploratory evidence against a mere
  threshold problem, not a final verdict.
- FC-VAE diagnostic job 354018 showed a finite first Adam step on all four
  ranks but extreme rank-local loss/gradient scales; the later failure remains
  unresolved.
- Public use now has four study-root commands: `download_data.py`,
  `prepare_data.py`, `run_experiment.py`, and `analyze_results.py`. Keep
  internal audit/tests behind this small interface rather than building
  another orchestration layer.
- Correction 2026-07-24: those four short commands are only wrappers over a
  21,414-line internal Python tree including tests. They are not the promised
  compact reference implementation. The target is a genuine five-file
  extraction (`download`, `prepare`, `models`, `run`, `analyze`) for one frozen
  source-faithful anchor, with the branch/evidence/DDP machinery retained
  separately as the forensic harness.
- Workflow reset 2026-07-24: `RUNBOOK.md` is now the canonical tutorial for
  every paper. The active Paper 1 route is a fresh PDF-derived `METHOD.md`,
  genuine five-file ISET implementation, tiny sanity run, then one full
  Table-III FC-SAE anchor before any additional infrastructure, publication,
  or exhaustive branch execution.
- Paper 1 source freeze completed 2026-07-24 at
  `studies/atk-2022-deep-autoencoder/METHOD.md` after fresh extraction and
  visual review of all 12 pages of PDF SHA-256 `f3098e...850f`. The declared
  first anchor is `P0-ISET-FCSAE`: all named residential meters; strict
  48-slot days; all-customer six-attack `M`; joint pre-split feature scaling;
  customer-disjoint B1/B2; printed test-set ADASYN; the full
  `48-400-300-200-100-100-200-300-400-48` sigmoid/Softmax FC-SAE; and printed
  threshold 0.58. Batch/epoch/convergence/seed choices are visibly labeled
  execution completions, not paper facts.
- Independent source re-audit 2026-08-11: the exact PDF hash and overall
  `METHOD.md` flow were reconfirmed after all 12 pages were visually inspected
  before opening the old reconstruction. Corrections/additions: Tables II/III
  have six, not seven, benchmark rows; VAE Eq. (9) mixes incompatible
  distributions/variables; VAE variance positivity and Algorithm-5 decoder
  input are undefined; precision prose conflicts with its formula; Table-II
  Naive Bayes F1 is arithmetically inconsistent; and neither Table II nor III
  admits one common prevalence from its DR/FA/PR rows under generous rounding.
  `P0` is explicitly a paper-primary `P+I` executable completion because
  printed Attack 3 is non-executable. The renewed source-freeze checkpoint is
  the next gate; do not audit code or resume compute before it is accepted.
- The fresh source pass independently reconfirmed pivotal non-uniqueness:
  Eq. (3)'s endpoint is impossible; “rows (customers),” all-customer `M`, and
  unseen-test-customer claims conflict; Algorithm 6 cannot calculate DR/FA
  from benign-only `X_TR` and its scalar width/layer loops cannot yield all
  Table-I layouts as written; Fig. 3's distinct latent width is absent; and a
  common Table-V model/common benign set mathematically requires invariant FA
  although the table reports attack-varying FA. CHECKPOINT 1 is now the only
  gate before the compact five-file implementation.
- CHECKPOINT 1 was approved 2026-07-24. The genuine compact implementation now
  exists at `studies/atk-2022-deep-autoencoder/reproduction/`: five direct
  files and 1,617 total lines, with no imports from the forensic `src/` tree.
  On real CER rows, `p0-tiny-v1` completed source verification, preparation,
  two FC-SAE epochs, scoring, metrics, baselines, Table V, and aggregation.
  The runtime FC-SAE has all printed widths
  `400,300,200,100|100,200,300,400` and 450,448 parameters. Tiny metrics are
  fixture-only (ACC 42.18%, AUC 42.23%); all six Table-V FA values are exactly
  65%, confirming the fixed-model/common-benign invariant. The next action is
  a new full P0 cache from raw verified archives, not the historical cache.
- The fresh full compact preparation materialized 2,251,290 strict benign
  profiles, 13,507,740 generated attack profiles, 1,500,520 customer-disjoint
  training profiles, and the exact 14,258,510-row printed `B2+M` population.
  Applying imbalanced-learn's default ADASYN to that population is not an
  ordinary preprocessing wait: with 48 features, sklearn selects brute-force
  neighbors and its first call entails about
  `750,770 × 14,258,510 = 10.7e12` profile-distance comparisons, followed by a
  second minority-only search. Preserve this as an exact-default
  executability result. The interrupted default call consumed 4,724.52 seconds
  wall time and 33,665.16 CPU-seconds (9.35 CPU-hours) without completing its
  first `kneighbors` call or producing `x_test.npy`. Job 373799 benchmarked 250
  exact queries on 16 CPU cores and linearly estimates 14.16 wall-hours for the
  two full neighbor searches alone. This is expensive but feasible overnight;
  it is not evidence that the authors could not have run ADASYN, whose library,
  hardware, and preprocessing boundary are unreported. Run the
  no-test-resampling interpretation explicitly
  as `I-ADASYN-NONE`, then a separately labeled scalable ADASYN sensitivity;
  never call either one the completed exact-default P0 cache.
- `prepare_data.py` now exposes opt-in paper-interpretation/corrected policies
  without changing historical defaults: four scaling scopes, printed versus
  absent anomaly-test ADASYN, pre-split versus training-only supervised
  ADASYN, and ISET all-customer versus B2-only malicious populations. Fixture
  tests prove corrected test sets contain no synthetic rows and supervised
  original customer/meter identities are disjoint. No full cache has yet been
  rebuilt under these new policies.
- SGCC preparation and both ordinary/DDP runners now expose all six frozen
  resolutions of 1,034 raw days versus 48 model inputs, all four missing-data
  readings, and customer-disjoint versus row-random sample splitting. Window
  IDs retain their source customer; fixture tests prove the customer-disjoint
  branch is disjoint and the row-random branch is not. These semantics are
  included in run fingerprints. Full rolling windows are structurally
  executable but must enter the frozen screening path rather than be mistaken
  for the historical full-vector result.
- Source-v2 recurrent builders execute both input layouts, both state-transfer
  policies, and repeat/first-step/autoregressive decoder schedules for
  LSTM-SAE/LSTM-VAE. A dedicated Algorithm-5 AEA decoder now feeds the prior
  scalar reconstruction back at every time step while recomputing attention
  from the prior decoder state; it executes concatenate/literal-sum merges,
  mirrored/top-only states, both input layouts, and both latent placements.
- The VAE runner now supports all frozen score IDs. Fixed-variance and learned
  decoder-variance-head branches calculate Monte Carlo multivariate Gaussian
  reconstruction density for 1/10/100 draws; raw probability is explicitly
  lower-is-anomaly, while MSE and MSE+KL remain higher-is-anomaly surrogate
  branches. Existing VAE results are still implementation-v1 surrogates.
- The ordinary runner now executes supplied printed constants and all three
  deterministic repairs of “median of IQR of ROC,” independently crossed with
  ISET-transfer/dataset-specific scope and B1-generated-attack,
  B2-validation-carve-out, or no-derivation populations. B2 validation rows
  are removed by identity from final test; SGCC transfer requires a frozen
  ISET threshold artifact. Fixed epochs, holdout/no-refit, holdout/all-B1
  refit, and five-fold-or-maximum-feasible cross-validation/all-B1 refit also
  change actual fit behavior and retain every history/timing. The DDP runner
  now has matching validation/refit/threshold semantics. Stable branch IDs
  resolve through the public runner into data, model, classical, validation,
  threshold, and Table-V arguments; preparation IDs are content-addressed and
  checked against cache metadata.
- ISET preparation now executes all registered Attack-1 factor scopes,
  Attack-2 half-hour/hour-pair granularities, all three minimal repairs of the
  non-executable Attack-3 interval, both hour-to-slot mappings, and all-4,225
  versus deterministic seeded-3,000 residential populations. The printed
  Attack-3 subtraction remains a non-executable evidence node. Attack
  regeneration is also explicit: fixed per data seed, per model seed, or per
  experiment index, with the resolved seed derivation stored in cache metadata.
- CER archive extraction now executes strict 1--48 days, trimming slots 49/50,
  duplicate-slot averaging, and 48-grid interpolation. ISET preparation also
  executes customer-disjoint and row/profile-random splitting; heldout attacks
  are bound by source-profile identity so row-random meters may overlap without
  mixing the actual profile rows. The existing 3.2-GiB cache remains the
  strict-day/customer-disjoint implementation-v1 artifact.
- Exact-ISET execution now uses that same `run_experiment.py` interface:
  `--dataset iset --table 3` trains Table III and derives Table V from the same
  persisted score vector; `--table 4 --sizes ...` trains the nested size cells.
  The 3.2-GiB cache reverified and loaded in 13.65 seconds locally, and an
  actual one-epoch FC-SAE fixture fit completed end to end. That historical
  cache is implementation-v1 and cannot be relabeled for source-v2 branches.
  The next gate is the content-addressed exact-ISET cache build followed by one
  real DDP smoke on the compute cluster, not a replacement matrix.
- The source-first visual map is the self-contained
  `site/papers/atk-2022-deep-autoencoder/index.html`, with a pointer at
  `studies/atk-2022-deep-autoencoder/PAPER_WORKFLOW.md`. It is a readable
  paper-order explanation, not an embedded Mermaid graph or a display of the
  internal branch combinatorics.
- Bounded local Gate-D evidence is
  `studies/atk-2022-deep-autoencoder/results/gate_d_bounded_sanity_20260724.json`:
  137 study plus 10 project tests pass; the real SGCC printed-anchor preflight
  is ready; the exact-ISET branch correctly stops at its missing
  content-addressed cache.
- Direct `--table 5` execution now covers common model/common benign,
  per-attack retraining, per-attack seeded benign resampling, and both,
  crossed with full-heldout/seeded-3,000 sizes. It persists all six score and
  identity sets and honors lower-is-anomalous VAE probability. The historical
  Table-III-coupled derivation remains only the common/fixed structural
  diagnostic.
- Eq. (10) visually prints squared L2 plus KL. Source-v2 FC/LSTM VAE builders
  now execute both `sum_squared_plus_kl` and the common
  `mean_mse_plus_kl` reading. With zero KL, their reconstruction terms differ
  by exactly the input dimensionality. Learned-decoder-variance branches use
  the analogous summed/mean Gaussian data term so the variance head receives
  gradients; that likelihood loss is a documented prose-consistent
  completion, while fixed variance plus summed squared error is the direct
  printed-loss branch.
- Exact-ISET seed 11 implementation-v1 cells (quarantined as reproduction
  evidence): FC-SAE full DR 22.50%, FA 37.13%,
  ACC 42.69%, AUC 42.59% (paper 81/15/83/81); FC-SAE half ACC 42.68%
  (paper 70); FC-VAE full 40.43/53.86/43.28/40.82%
  (paper 88/11/88.5/85). FC-SAE score correlates 0.99945 with input energy;
  even reversed test-label oracle ACC is only 57.72% on primary rows. These
  are strong one-seed exploratory non-reproductions, not a final verdict.

## Working patterns

- Run deterministic tests with `bash scripts/test.sh`; it supports the root environment and the legacy local environment.
- Preserve raw files in place and identify them by checksum; study artifacts belong under `studies/<study-id>/results/`.

## Don't repeat

- Do not substitute 48-day SGCC windows for the paper's 48 half-hour CER profiles when assessing the primary reproduction hypothesis.
- Do not let literature provenance work displace the exact-data, paper-literal reproduction task.
- Do not silently correct the paper in the primary track; corrections belong in a separately labeled controlled analysis.
- Do not report a best/lucky seed as a reproduced result.
- Do not search Keychain metadata, browser storage, shell history, credential files, or broad home-directory locations for dataset access. A broad credential audit likely triggered workplace endpoint protection on 2026-07-21. Restrict work to the project, explicit dataset locations, and user-supplied authorization.

## Open questions

- The design for the exhaustive confirmatory phase is written and visible at
  `docs/plans/2026-07-22-confirmatory-branch-sweep-design.md` (anchored branch
  enumeration, AUC screening funnel, controls, mechanism demonstration). It is
  a draft: freeze checklist + user CHECKPOINT required before any execution.
- Exact membership and order of the three-paper core corpus after Paper 1.
- Reproduction tolerances, finite hyperparameter envelope, seed count, split policy, and computational stopping rule must be frozen before confirmatory runs.
- Exact ScienceDB CER archives and the named semantic allocation branch have
  passed the implemented code gate, preparation, execution preflight, and
  bounded model smoke. Tables III--V now need the compute cluster result cells.
- Public canonical repository: <https://github.com/fjoad/atk-evidence>, default branch `main`.
- The user explicitly authorized an end-to-end exploratory reconstruction of Paper 1 Tables I-V on 2026-07-21, including timed runs and documented author-intent assumptions. It is not retrospectively preregistered confirmatory evidence.
- Exact ISET execution requires seven restricted official files: six consumption archives plus the SME/residential allocation file; login alone does not grant access.
- The post-verdict controlled solution is specified at
  `docs/plans/2026-07-23-paper-1-controlled-solution.md`. It is deliberately
  gated until Tables I--V, confirmatory assessment, and the Paper 1 LaTeX
  verdict are frozen.

## User emphases

- 2026-08-23 clean-reader rebase: preserve a stateful plan before doing more
  work. The governing flow is complete paper orientation → disposable sandbox
  → return to the paper and freeze one reasonable-reader completion → assess
  the existing five-file implementation → run or admit one exact-data anchor →
  inspect the complete numerical result → promote only scientifically necessary
  `N`, `M`, or `A` depth. The plan must show the current step, preserve
  loop-backs, and prevent future agents from resuming the former numerical
  branch sequence automatically.
- 2026-08-20
  [`three-part evidentiary frame`](decisions/2026-08-20-three-part-evidence-frame.md):
  assess Paper 1 separately on
  **numerical reproduction** (whether the reported result is reproduced),
  **mechanism identification** (whether added component `Z` demonstrates and
  uses the capability claimed to explain model `B`'s advantage over `A` on
  structure `S`), and **attainability** (whether the target lies outside the
  observed performance envelope with no trend suggesting ordinary additional
  search will close the gap). Treat these as distinct claims requiring distinct
  evidence; do not infer the mechanistic or attainability conclusion from a
  numerical non-reproduction alone. Begin investigation in a small disposable
  discovery sandbox to find a capability-discriminating question, then freeze
  and rerun that question through the formal evidence path; sandbox outcomes
  remain exploratory and cannot select paper interpretations post hoc.
  “Breadth first” means many cheap, small, question-specific sanity probes over
  competing explanations before depth; one costly full run per model family or
  thousands of lines of branch support is already execution depth, not the
  intended first breadth layer.
- 2026-08-11 baseline-first execution: before testing alternative assumptions,
  run one frozen straight-through Paper-1 anchor end to end. The current runnable
  lane is full ISET, FC-SAE, seed 11, batch 512, original `B2+M`, producing the
  Table III row, full-data Table IV cell/timing, and Table V attack rows from one
  model. It is explicitly `I-ADASYN-NONE`, not completed printed `P0`; the
  multi-trillion-pair default test-ADASYN failure remains visible. Do not start
  an ambiguity sweep before inspecting this result.
- 2026-08-11 non-executable reporting rule: every paper statement that cannot
  exist or run must be visibly identified on the study site and in the LaTeX
  report. Preserve the literal failure, predeclare all materially reasonable
  executable repairs, run them under separate `I` IDs, and show every result
  beside the reported target. A repair is never relabeled as the literal method;
  a matching repair must be reported as readily as a non-match.
- 2026-08-09 reframe: extracting the paper's actual method is the most rigorous
  and reasoning-intensive part of every study. Implementation should then be a
  small transparent transcription. Keep global infrastructure minimal, prove the
  measuring path cheaply, run one watched full anchor, and only then scale. The
  shared research documentation must preserve this across compaction;
  see `docs/decisions/2026-08-09-paper-first-minimal-instrument.md`.

- The honest hypothesis is that the reported numbers will not be reproducible from the papers as written, but the project must be genuinely open to being wrong.
- Exact paper-described algorithms and procedures are the highest priority; add nothing extra to the primary track.
- 2026-07-23: a prior contract, passing test, or registered ambiguity is never
  sufficient proof of fidelity. Before compute, reconstruct the method from the
  PDF alone and require claim-to-code-to-cache traceability; quarantine results
  immediately when that chain fails.
- 2026-07-23 visual architecture correction: Table I/Section IV-C require all
  encoder hidden widths and the full mirror, which invalidates
  implementation-v1 FC-SAE and FC-VAE. Figs./prose depict latent layers, while
  Algorithms 2/5 directly reuse the terminal encoder state or attention
  context; LSTM-SAE and AEA therefore remain quarantined algorithm-literal
  branches, with distinct-projection branches also required. LSTM-VAE's hidden
  structure is aligned, but latent width and probability score remain
  unresolved.
- 2026-07-23: the user requires three separate families for every material
  issue: the printed method even when statistically wrong, every defensible
  interpretation of ambiguous/contradictory text, and the scientifically
  corrected method. “All” means a documented finite coverage closure against
  every material PDF statement and omission, not an unbounded claim over
  imaginable code.
- 2026-07-23 standing authorization: continue implementing that approved
  `P`/`I`/`C` mandate autonomously through safe structural and fixture-test
  gates. Do not repeatedly ask permission for each ambiguity branch. Stop only
  at a declared compute/freeze checkpoint, an external blocker, or an action
  requiring materially new authority.
- Paper 1 branch-lattice v1 is machine-readable at
  `config/branch_lattice.toml`: 22 model/data families, 22 printed anchors, 899
  interpretive configurations, 22 separate corrected controls, and 2,763
  three-seed screening attempts. Point screening estimate is 558.7 GPU-hours
  plus 57.4 CPU-hours, or 49.7 ideal active GPU-job hours at the three-job cap
  (99.4 h under the 2x runtime factor), excluding queue time. The
  52.57-billion arbitrary Cartesian product is explicitly excluded; every
  option and compatible option pair is verified, and all 36 ambiguity-register
  rows have machine-checked coverage references. Threshold formula and
  threshold scope are independent dimensions, while impossible formula/label
  and printed-constant/dataset-specific combinations are machine-excluded.
  Algorithm 6 now has three
  explicit branches: literal uniform-width search (36 evaluations), per-layer
  coordinate search capable of unequal widths (86), and direct Table-I replay;
  the earlier eight-evaluation budget had no paper basis and was removed.
- 2026-07-24: SGCC has no six attack-type labels, so the seven-class
  “multiclass SVM” reading applies only to ISET. The SGCC family remains binary
  and fails loudly if seven-class labels are requested. This source-bound
  correction was frozen before replacement results; see
  `docs/decisions/2026-07-24-sgcc-multiclass-label-scope.md`.
- Ambiguities may use reasonable assumptions only with complete documentation.
- Evidence must come from rigorous experiments and statistical assessment, aiming for the strongest defensible conclusion.
- Deliver one LaTeX-style rebuttal/reproduction report per paper and a combined report.
- After Paper 1 reconstruction and analysis are finished, separately design
  the method we would actually use to solve electricity-theft detection,
  explain and isolate why the literal method fails, and test whether the new
  method honestly exceeds the published results. Do not begin this early or
  blur it into reproduction evidence.
- Claims should be supported by independently rerunnable proof-quality evidence; data acquisition and setup must be explicit from a fresh public clone.
- Keep cluster orchestration minimal: short
  `sbatch` wrappers around the Python programs, without a separate manifest,
  probe framework, packed-worker layer, or automatic scheduler.
- Keep the public explanation and reference implementation proportionate. The
  exhaustive audit harness may remain large, but it must not be confused with
  the amount of code required to implement one paper-defined experiment.
- 2026-07-24: the user explicitly identified that project effort had become
  inverted—implementation infrastructure and documentation dominated while
  eligible experiments lagged. For every paper, reading/source freeze should
  dominate reasoning and experiments should dominate elapsed time. Stop if
  code/framework growth is delaying the first eligible full result.
- Publish one GitHub Pages site for the repository, with one path and one
  scientific PDF per paper plus a later synthesis. The public site must deploy
  from a dedicated static directory so internal docs, local data, and paper
  PDFs are not exposed.
- 2026-07-22: user confirmed batch-512 stays the frozen primary (stated motive:
  GPU utilization); batch 32 remains the declared sensitivity. LSTM-AEA is
  planned under the batch-32 branch via DDP (local batch 8 fits ~6.4 GiB).
  `run_model_ddp.sbatch` now accepts an optional config plus direct runner
  arguments, but no replacement compute is authorized before Gate C closes.
- the compute cluster 2026-07-23: no queued or running project jobs. Successful Table II
  cells = 20/33 (classical 12, FC-SAE 3, supervised feed-forward 3, LSTM-VAE
  1, LSTM-SAE 1); FC-VAE failed 3 and 10 cells are unrun. At the user's
  direction, the single-T4 batch-32 LSTM-SAE sensitivity was cancelled after
  23:27:30; it is resource evidence only.

## When to update this file

Update inline when a non-obvious environment fact, failed path, user emphasis,
or active decision appears. Keep entries terse and prune beyond roughly 200
lines. Promote durable causal corrections to `EVIDENCE-AND-LEARNINGS.md`.
