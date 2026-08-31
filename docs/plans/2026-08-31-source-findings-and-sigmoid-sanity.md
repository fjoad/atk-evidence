# Publish the source checks, then investigate Sigmoid cheaply

Date: 2026-08-31

User approval: publish the discussed findings with clear questions inside the
relevant methods sections, then investigate Sigmoid with quick sanity checks
rather than a full retraining run.

This bounded follow-up sits inside Phase 7 of the clean-reader plan. It does
not authorize a new paper, full training, a seed sweep, or a broad search.

## Sequence and current step

1. **COMPLETE — Publish existing evidence.** Extend the current report with
   the source review, three normalization limits, Sigmoid-range sensitivity,
   and reversed-direction counterevidence. Preserve the original numerical
   result and keep original-row checks distinct from the full resampled test.
   Put the plain-language question before each calculation. Update the home
   page and README briefly; link frozen sources and unmodified result records.
   Test the site, publish through existing GitHub Pages, and verify live bytes.
   All 217 deterministic tests passed (140 study, 77 root), including 17
   public-report checks; the local report response matches its source bytes.
   The original numerical/scientific records and implementations are unchanged.
   Published revision `dc37bbed81323369adf35ab450c7d3138601a039`; Pages run
   `33419150100` succeeded. All nine checked public pages/assets returned 200
   and matched local source bytes, including the source-assumption sections.
2. **COMPLETE — Freeze the next cheap question.** Specify an exploratory `X/A`
   Sigmoid check using the unchanged original data preparation. Begin with a
   no-training bound on the complete prepared evaluation, including synthetic
   benign rows. Record whether any learned check can add information after
   that result. Fix the sample, controls, time cap, promotion and stopping
   rules before scoring. Use a cluster compute node only.
   The exact no-training setup is now in
   [SIGMOID_SANITY.md](../../studies/atk-2022-deep-autoencoder/SIGMOID_SANITY.md):
   complete bound plus clipped-input and constant-half controls, both directions,
   512-row pilot then at most one full pass; 240-second process cap. No fitted
   comparison is part of this first wave.
   All 223 deterministic tests passed (140 study, 83 root), including six
   new hand-sized Sigmoid fixtures. Scientific code is 182 direct lines and
   reuses the unchanged geometric helpers.
3. **IN PROGRESS — Run only the promoted checks.** No full retraining. A direct
   substitution in saved Softmax weights is not a trained Sigmoid model and
   cannot by itself establish Sigmoid failure. Any small fitted comparison
   must remain clearly exploratory, paired, and time-limited; do not tune on
   its evaluation labels or escalate automatically to the complete dataset.
4. **PENDING — Save and discuss.** Preserve every result and runtime, update
   the explanation register and working memory, then stop for discussion.
   The user's permission to publish the already discussed source checks does
   not authorize publishing new Sigmoid outcomes before discussing them.

## What the next checks must distinguish

- Does changing only the output range leave the complete reported target
  outside the bound on the actual prepared evaluation?
- If the bound permits it, does a feasible, label-blind reconstruction or a
  tiny fitted model show a useful route, rather than just a label-aware oracle?
- Does reversed scoring change the answer, and which explicit paper choices
  have then been changed?

A passing upper bound is not evidence that training reaches it. A short
training failure is not an impossibility result. The question and stopping
rules matter more than running another model.

## Verification and publication boundary

Run deterministic tests, check local links and conservative rounding, and
preserve original records and scientific files. Commit the approved site
update and publish it before starting experimental scoring. Record the exact
deployed revision. Later experimental records may be committed locally but
must not be pushed or added to the website until discussed.
