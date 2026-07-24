# Publication and reference simplification

**Created:** 2026-07-24
**Status:** Publication layer implemented; reference-code extraction pending

## Problem

Paper 1 currently exposes four small commands over a 21,414-line internal
Python tree (including tests). That tree is useful for ambiguity coverage,
immutable evidence, and cluster execution, but it is not the compact
implementation the user requested and must not be presented as such.

The embedded Mermaid workflow also failed the readability requirement. The
requested artifact is a polished standalone file that explains the paper in
the paper's own order.

## Delivered in this pass

1. A self-contained Paper 1 HTML method map.
2. A minimal multi-paper project landing page.
3. A root README that describes the multi-paper scope and the current code-size
   truthfully.
4. A conventional Paper 1 LaTeX report scaffold.
5. A minimal GitHub Pages workflow that publishes only `site/`.

## Five-file reference target

The eventual reference implementation will contain exactly these
researcher-facing responsibilities:

1. `download_data.py`
2. `prepare_data.py`
3. `models.py`
4. `run_experiment.py`
5. `analyze_results.py`

It must be a real, readable extraction of one frozen source-faithful anchor,
not wrappers over the full branch engine. The existing forensic harness remains
available for ambiguity sweeps and evidence verification.

## Safety boundary

No current evidence code is deleted or moved in this pass. That refactor can
change imports, fingerprints, and artifact eligibility, so it requires a
separate tested migration after the source-v2 anchor and cache contract are
stable.

## Finish conditions

- The visual opens as one local file and needs no build step.
- The README cannot be mistaken for a single-paper project.
- Pages exposes one site with one path per paper.
- The LaTeX source compiles reproducibly before a PDF is published.
- The compact reference track is measured independently from the forensic
  harness and is not declared complete until its scientific outputs match.
