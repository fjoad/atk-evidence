# Input data — provenance and preparation

Everything needed to rerun this study from scratch. The raw series is **not**
committed here: it is a third-party benchmark distribution, and redistributing
it would put this repository in the position of relicensing someone else's data.
Obtaining it takes a few minutes, and the checksums below let you confirm you
have byte-identical inputs to ours.

## What the study needs

Exactly two files:

| File | Shape | Content |
|---|---|---|
| `processed_clean_scada_dataset.csv` | 8,761 × 31 | Hourly benign SCADA readings for the C-Town network |
| `processed_scada_adj_matrix.csv` | 31 × 31 | Node adjacency for the same 31 sensors |

## Checksums of the exact inputs used

```
b9da65a0ce883f872711f14e2c844ca45f1347112ee26c17cb91ec3a1353ee99  processed_clean_scada_dataset.csv
427679e04b469e268f54f1232eeb5a9866bbab361841364d08bc5686da83717e  processed_scada_adj_matrix.csv
```

Verify with `shasum -a 256 <file>`. Every result in this study was produced from
files matching these digests, and each recorded run logs them.

## Provenance

The underlying network is **C-Town**, the benchmark distribution network from
the BATADAL competition (Battle of the Attack Detection Algorithms). The benign
hourly series is the clean SCADA dataset distributed with the DeepH2O work built
on that benchmark. The audited paper states it generates its own 1,400-hour
series with EpanetCPA; that series is not published, so this study uses the
canonical benign C-Town series instead. The physics is the same network and the
same simulator family; the exact readings differ. This is recorded as a
limitation in the report rather than glossed over.

## Expected format

`processed_clean_scada_dataset.csv` — header row, then one row per hour:

```
L_T1,L_T2,...,L_T7,F_PU1,...,F_PU11,F_V2,P_J280,P_J269,...,P_J422
3.9962,1.1183,...,0.0,...,  ...
```

Column order matters and is asserted in code (`data.FEATURE_LABELS`): 7 tank
levels (`L_T*`), 11 pump flows (`F_PU*`), 1 valve flow (`F_V2`), 12 junction
pressures (`P_J*`). Prefixes are stripped when matching labels.

`processed_scada_adj_matrix.csv` — **no header**, 31 rows × 31 columns of 0/1.
The distributed file is directed and carries self-loops; the code symmetrizes it
and removes the diagonal, yielding **44 undirected edges**. Do not pre-symmetrize
it yourself — `data._benign_and_adj()` does this and the edge counts in the
report assume it.

## Pointing the code at your copy

```bash
export WATER_DATA=/path/to/your/data     # directory holding the two CSVs
python src/run.py --help
```

`WATER_DATA` defaults to `~/data/BATADAL/DeepH2O`.

## Sanity check before trusting any result

```bash
python src/test_detect.py     # detector regression tests, must pass
python src/sanity.py --size 31
```

`sanity.py` prints attack visibility, single-sensor separability,
zero-parameter detector scores, per-attack-class ceilings, and how much
spatial/temporal structure the data actually contains. If those numbers differ
materially from the report, your inputs differ from ours — check the digests
above before interpreting anything downstream.

## Attacks are generated, not downloaded

Malicious samples are not part of the input. They are synthesized from the
benign series by the three equations the paper prints (replay, denial of
service, data manipulation) in `data._build_observed()`. The attack schedule is
deterministic given the seed, so any run is reproducible from the benign inputs
alone.
