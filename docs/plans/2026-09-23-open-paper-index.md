# An open-ended paper index

Remove the fixed paper switcher and collection-size labels. The homepage is
the growing index; each paper page has a simple link back to it. Remove
fixed-count language from homepage metadata, navigation and renderer code.
Keep paper titles, publication years, section anchors and research unchanged.

- [x] Update navigation, index wording/style and renderer.
- [x] Test count-independent rendering, links and the existing claim checks.
- [ ] Check the browser, commit, publish the approved site follow-up and
  verify the deployed version.

No new experiments or changes to the earlier studies' evidence.

All 32 focused notebook/report/registry tests pass. A constructed additional
registry entry leaves existing article navigation unchanged. No fixed label
map, tabs or collection count remains in the renderer. Scientific result
bytes, assumptions and published section anchors are unchanged.

The first live check exposed cached pre-change CSS with the new unnumbered
index markup. The renderer now fingerprints the stylesheet URL for the index
and all articles. Tests require the content-derived version on every main
page, preventing an old cached stylesheet from keeping the removed layout.
