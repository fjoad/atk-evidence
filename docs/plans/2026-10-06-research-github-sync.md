# Synchronize research records with GitHub

The user paused experiments and requested that research code, results, and
writeups be current on GitHub. Website work is outside this request.

1. Verify local and remote state; push the 42 committed research changes
   through `be09ed0` without rewriting their history. Completed.
2. Refresh repository, study, documentation, and report entry points that
   still describe older research stages. Link the existing audited records;
   do not change findings, raw artifacts, or frozen contracts. Completed.
3. Check the documentation links and diff, commit and push the updates, then
   verify GitHub CI and that local and remote `main` match.

The first push's CI run, `37445679293`, failed one source-audit test: its
current METHOD hash was pinned to `d47a6de`, before `be09ed0` added links to
the sequential contracts. The original audit hash and numerical results are
unchanged. Update only the test's current-document pin to `be09ed0` and rerun
the required checks; do not rewrite the historical audit to satisfy the test.

Local validation passes: the six source-audit tests reproduce the failure
before the pin correction and pass afterward; the repository suite runs
450 cases with 405 passes and 45 environment skips. All 169 relative links
in the changed writeups resolve, the edited diff has no whitespace errors,
and the existing journal consistency check passes without modifying the site.
GitHub Actions records the final pushed revision's CI result.

No experiment, new fit, seed, cluster allocation, website regeneration, or
final scientific report is part of this synchronization. The next scientific
decision remains the separately specified and timed tanh CER pilot.
