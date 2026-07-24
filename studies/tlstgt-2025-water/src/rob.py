"""Robustness sweep: try to INFLATE our STGT result toward the paper's by varying
every ambiguous/generous interpretation. If STGT-31 never approaches the paper's
F1=76.3 (and the size trend never appears), our low numbers are not a suppression
bug -- they are the honest result.
"""
import numpy as np, data, models, detect, run


def stgt_eval(size=31, block=60, frac=1.0, fa=5.0, train_all=False, epochs=15, seed=0,
              delta_scale="raw"):
    A = data.graph(); idx = data.figure_nodes(size)
    ds = data.make_datasets(idx, seed=seed, block=block, attack_frac=frac,
                            delta_scale=delta_scale)
    a = models.normalized_adj(A[np.ix_(idx, idx)])
    m = models.STGT(data.WINDOW, size)
    if train_all:                                   # literal reading: train on the 50/50 set
        m = models.train(m, ds["hist"][ds["train"]], ds["target_obs"][ds["train"]], a, epochs=epochs)
    else:                                           # benign-only (generous for detection)
        tb = ds["train_benign"]
        m = models.train(m, ds["hist"][tb], ds["target_true"][tb], a, epochs=epochs)
    r = lambda i: run.resid_forecaster(m, ds, i, a)
    return detect.evaluate_forecaster(r(ds["train_benign"]), r(ds["val"]), ds["labels"][ds["val"]],
                                      r(ds["test"]), ds["labels"][ds["test"]], fa_target=fa,
                                      val_clean=ds["win_clean"][ds["val"]])


CONFIGS = [
    ("baseline  block60 all-sensor benign FA5", dict()),
    ("BRANCH: eq4 delta in per-sensor sigma", dict(delta_scale="sigma")),
    ("subset 40% sensors (cross-sensor incons.)", dict(frac=0.4)),
    ("short bursts block=15", dict(block=15)),
    ("high false-alarm budget FA=20%", dict(fa=20)),
    ("literal 50/50 training", dict(train_all=True)),
    ("MOST GENEROUS subset0.4 block15 FA20", dict(frac=0.4, block=15, fa=20)),
    ("+ different seed", dict(frac=0.4, block=15, fa=20, seed=7)),
]
print("STGT @ 31 nodes   (paper reports: F1=76.3  ACC=84.8  DR=74.7)")
for name, cfg in CONFIGS:
    m = stgt_eval(size=31, **cfg)
    print(f"  {name:44s} F1={m['F1']:5.1f}  ACC={m['ACC']:5.1f}  DR={m['DR']:5.1f}  FA={m['FA']:4.1f}")

print("\nSize trend under MOST GENEROUS  (paper STGT F1 climbs 52.3 -> 59.4 -> 76.3):")
for s in (10, 20, 31):
    m = stgt_eval(size=s, frac=0.4, block=15, fa=20)
    print(f"  size {s:2d}: F1={m['F1']:5.1f}  DR={m['DR']:5.1f}")
