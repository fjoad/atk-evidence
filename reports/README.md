# Scientific reports

ATK Evidence will publish:

- one standalone report for each registered paper at
  `reports/<study-id>/main.tex`; and
- one cross-paper synthesis at `reports/synthesis/main.tex` after the
  individual papers' three findings are frozen.

The reports use conventional scientific sections: claim and scope, source
protocol reconstruction, discovery and question selection, methods,
fidelity/ambiguity audit, experimental results, statistical assessment,
limitations, and bounded conclusions. Raw data and copyrighted paper PDFs are
never embedded.

Every paper report must distinguish three findings:

1. **Numerical:** whether the reported pattern was recovered under the finite,
   predeclared paper-consistent space.
2. **Mechanistic:** whether capability-sensitive tests identify the claimed
   component or structure as the cause of the advantage.
3. **Attainability:** where the target lies relative to the declared empirical
   performance envelope and whether observed trends give ordinary additional
   search a credible route to it.

The combined conclusion may interpret the intersection of those findings, but
none substitutes for another. An empirical envelope is not an infinite-space
proof, and no report infers author intent or an undocumented implementation.

Exploratory sandbox results may explain why a formal question was selected.
They remain labeled exploratory and cannot be presented as confirmatory
evidence. Reports should show the causal claim map (for example, `B > A`
because `Z` exploits `S`), the competing predictions, all material failures,
and the boundary of every search or static argument.

## Build

The local build uses [Tectonic](https://tectonic-typesetting.github.io/):

```bash
mkdir -p reports/atk-2022-deep-autoencoder/build
tectonic --outdir reports/atk-2022-deep-autoencoder/build \
  reports/atk-2022-deep-autoencoder/main.tex
```

The current macOS Homebrew Tectonic install crashes in its system-configuration
layer on this workstation before TeX compilation begins. No PDF from that
failed invocation is published. The source remains the artifact until it
passes a clean local or CI compilation.

Generated PDFs can be copied into `site/reports/` when they are ready for
publication. The GitHub Pages workflow deliberately deploys only `site/`, so
internal notes, local data, and paper PDFs cannot be exposed accidentally.

## Publication layout

```text
https://fjoad.github.io/atk-evidence/
  papers/<study-id>/       readable method map for one paper
  reports/<study-id>.pdf   frozen scientific report
  reports/synthesis.pdf    combined report after all paper findings
```

There is one GitHub Pages project site for this repository. Each paper receives
its own URL path within that site.
