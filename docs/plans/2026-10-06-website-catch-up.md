# Bring the public website up to the saved research checkpoint

The user requested a complete website update after research synchronization
commit `8f616c2`. This authorizes updating and publishing the existing GitHub
Pages site. It does not restart research, change scientific results, or
reassess the older water and autoencoder studies.

1. Reconcile the latest study overview, result records and explanation
   register with the public journal. Preserve earlier section anchors and
   make their historical status clear where subsequent results supersede them.
2. Extend the poisoning investigation through the completed GRU, standalone
   AEA bounds and corrections, sequential learning checks, research pair,
   saved-state diagnostics, and full-length tanh learning check. Update the
   opening summary, coverage, next question and final conclusion. Link source
   records and distinguish constructed learning from research-data evidence.
3. Refresh the index and repository publication pointers. Check every paper's
   current summary and preserve the older studies' awaiting-reassessment
   notices and linked archives. Keep the growing index, All papers links,
   undated reading flow and existing visual style.
4. Render all journals; verify links, anchors, figures, numerical summaries,
   desktop/mobile layouts, repository tests and data verification. Record
   the outcome in STATUS and CONTEXT, then commit and push.
5. Confirm GitHub CI and Pages deployment for the published revision, compare
   live files with the committed pages, and inspect the live navigation and
   updated conclusion. No experiment is part of this work.

Status: steps 1–4 complete locally; publication and live verification pending.

The public account covers every completed detector pair and the later AEA,
interface, saved-state, gate and recurrent-cell checks. The source records
and older research accounts are unchanged. Both PNG figures match their
originals byte-for-byte; the renderer now resolves source-relative image
paths to their public paths. All 65 existing paper-page anchors survive.

Validation: 34 focused site/registry/report checks pass; the complete
bootstrap/test workflow runs 452 cases, with 407 passes and 45 environment
skips. Strict data verification selects the documented
`sciencedb-csv-semantic-equivalence-v1` branch; original restricted TAB access
is not claimed. All 99 rendered GitHub main-file links resolve locally.
Desktop and 390-pixel mobile checks verify figures and table overflow;
the page itself has no horizontal overflow. The older study source journals
are byte-identical to `8f616c2`.

Environment-only verification failures are preserved in
`/tmp/atk-site-20261006.Cnba28`: bootstrap initially lacked pip, repaired with
ensurepip; the default MPS run failed on unsupported QR operations. The
successful full run uses `KERAS_TORCH_DEVICE=cpu`. No scientific model or
result changed to satisfy the checks.
