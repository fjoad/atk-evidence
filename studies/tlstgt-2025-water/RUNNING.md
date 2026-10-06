# Running the water-network study

```bash
export WATER_DATA=/path/to/data        # see DATA.md
python src/test_detect.py              # detector regression tests
python src/sanity.py --size 31         # breadth checks; trains nothing
python src/forensics.py                # forensics on the reported table
python src/protocol_search.py          # exhaustive protocol search
python src/run.py --help               # the full model comparison
```

`run.py` exposes every pre-registered ambiguity axis as a flag: `--delta-scale`,
`--train`, `--thresh`, `--transformer`, `--errfit`, `--adj`, `--window`,
`--ablate`, `--batch-s`. Defaults are the paper's literal reading.

**Reporting rule:** always state which configuration produced a number. Training
on the paper's literal 50/50 split and training on benign windows only differ by
roughly 25 F1 points, so an unattributed figure is meaningless.

## Files

`src/data.py` inputs, Figure-1 graphs, attack synthesis · `src/models.py` the
nine detectors · `src/detect.py` residual → Mahalanobis → threshold → metrics ·
`src/run.py` orchestration · `src/sanity.py` zero-parameter checks ·
`src/depth_auc.py`, `src/depth_replay_capacity.py` aimed probes ·
`src/forensics.py`, `src/protocol_search.py` analysis of the reported table ·
`src/test_detect.py` regression tests · `results/` raw outputs.
