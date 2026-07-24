# Scientific reports

ATK Evidence will publish:

- one standalone report for each registered paper at
  `reports/<study-id>/main.tex`; and
- one cross-paper synthesis at `reports/synthesis/main.tex` after the
  individual verdicts are frozen.

The reports use conventional scientific sections: claim and scope, source
protocol reconstruction, methods, fidelity/ambiguity audit, experimental
results, statistical assessment, limitations, and bounded verdict. Raw data
and copyrighted paper PDFs are never embedded.

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
  reports/synthesis.pdf    combined report after all paper verdicts
```

There is one GitHub Pages project site for this repository. Each paper receives
its own URL path within that site.
