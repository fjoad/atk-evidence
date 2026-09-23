# Continuous website narrative

The user wants a continuing investigation, not a date-by-date diary. Remove
calendar prefixes, update-date banners and dated editorial introductions from
the main three-paper pages. Keep publication years for paper identification,
the investigation's order, scientific claims/corrections and linked source
records. Preserve published section anchors so old links still work.

- [x] Adjust the renderer and visible prose; keep internal provenance dates.
- [x] Add a regression check for undated headings and preserved anchors.
- [x] Check generated pages, links, tests and browser navigation.
- [ ] Commit, publish this follow-up to the approved site, and verify it live.

No experiment, result or archived technical record changes.

The 31 focused notebook/report/registry tests pass. The renderer strips dated
heading prefixes only from display text, keeping existing IDs and in-page
links. Browser review confirms the undated contents list and page header;
all scientific display/link tests remain passing. Publication years remain
as paper identification, not as a timeline for the investigation.
