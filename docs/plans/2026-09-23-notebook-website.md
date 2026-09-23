# Three-paper notebook website

**Date:** 2026-09-23. **Scope:** writing, site structure, verification and
user-authorized publication to the existing GitHub Pages site. No experiments.

The homepage should be a quiet index of the three registered papers. Each
paper gets a consistent notebook: what it claims, starting hypothesis,
chronological investigation/results/corrections, supporting records, and a
bounded current conclusion at the end. Preserve old detailed accounts and
their URLs where possible; do not turn earlier provisional results into new
or stronger evidence. Electricity theft/deep autoencoder are one paper, not two.

- [x] Inspect current sources and evidence; identify corrections and provenance.
- [x] Build a small shared journal renderer and understated responsive design;
  make the homepage only an introduction and three full-title paper links.
- [x] Reconstruct dated notebooks for Papers 1/2 from preserved records; label
  retrospective entries. Keep older reports/notes accessible as archives.
- [x] Extend Paper 3 with the mathematical checks, averaging caveat, competing
  explanations, supportive pilot results, untested ensemble and final conclusion.
- [x] Verify numeric claims, source/archive links, metadata, generated output,
  mobile/desktop navigation and layout, and the relevant/full software tests.
- [ ] Update handoff documentation, commit, publish to GitHub Pages, verify
  the deployed pages, and report the finished result. Do not start model work.

Publication was explicitly authorized through the user's follow-up selection.
All restricted datasets, original source PDFs, models and raw score arrays
remain excluded. Existing research contracts and result bytes stay unchanged.

## Verification before publication

All three notebooks are generated from their study RESEARCH_LOG.md sources.
The old command remains a compatibility wrapper. Five new site tests enforce
the small homepage, three tabs, source generation, conclusion-at-end structure,
reassessment notices and key source/experimental caveats. Existing scientific
display checks now target Paper 1's notebook instead of requiring its results
on the homepage. The full suite passes 344 tests with 10 environment-specific
skips; the focused site/registry suite passes 30 tests. No model or data
experiment ran. Existing study results, scientific code and contracts are
unchanged in this editorial diff.

Browser checks covered the desktop homepage, all three paper tabs, expanded
contents, the new mathematics entry, archive/back navigation, and a 390px
mobile viewport. Horizontal overflow stays inside table regions, not the page.
The viewport override was reset. Static checks validate local links/anchors,
source links, metadata, all existing scientific report tables and output figures.

The user clarified that the older water/deep-autoencoder investigations were
done with an earlier model/workflow and should be rethought/redone later, one
paper at a time. Their results are preserved; the new notebooks explicitly say
they await reassessment and do not claim fresh validation.
