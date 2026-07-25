"""Run the paper-literal TL-STGT reproduction and print our Table I vs reported.

Usage:
  OMP_NUM_THREADS=1 python run.py                          # paper-literal (raw delta, S=32)
  OMP_NUM_THREADS=1 python run.py --delta-scale sigma      # documented delta branch
  OMP_NUM_THREADS=1 python run.py --batch-s 1 32           # evaluate at several batch sizes

Graphs are the exact Figure-1 node sets (data.figure_nodes). Shallow models classify
the observed reading; forecasters (FFNN/LSTM/TGCN/STGT) detect via Mahalanobis on the
forecast residual; AE via reconstruction error. Test/val are 50/50 by construction.

S (the detection batch size) only affects DETECTION, not training, so each model is
trained once per delta branch and then evaluated at every requested S.
"""
from __future__ import annotations
import argparse, sys, numpy as np
import data, models, detect

SIZES = [10, 20, 31]
FORE = ["ffnn", "lstm", "ae", "tgcn", "stgt"]
GRAPH = {"tgcn", "stgt"}
# Zero-parameter reference points. They are NOT from the paper; they exist to
# measure whether the task needs a model at all. They run through the identical
# residual -> Mahalanobis -> threshold -> metric path as every trained model, so
# any difference is attributable to the model rather than to the evaluation.
TRIVIAL = ["persist", "zscore"]
NAME = {"svm":"SVM","rf":"RF","lgbm":"LGBM","ffnn":"FFNN","lstm":"LSTM","ae":"AE",
        "tgcn":"TGCN","stgt":"STGT","persist":"PERSIST*","zscore":"ZSCORE*"}
# STGT trained from scratch for the same TOTAL epochs the transfer chain uses,
# so a transfer "gain" cannot be explained by extra training budget alone.
BUDGET_MATCH_NAME = "STGT-3x*"
ORDER = ["ZSCORE*","PERSIST*","SVM","RF","LGBM","FFNN","LSTM","AE","TGCN","STGT","STGT-3x*","TL-STGT"]
REPORTED = {
    "SVM": ([6.1,13.2,26.2],[80.7,78.4,77.7],[3.2,9.3,20.1]),
    "RF": ([17.5,22.7,28.9],[82.4,80.1,76.1],[9.5,12.4,30.1]),
    "LGBM": ([19.8,20.1,22.3],[82.5,81.3,81.9],[11.1,11.3,13.3]),
    "FFNN": ([20.9,24.1,36.1],[78.6,80.6,83.3],[14.5,22.7,24.1]),
    "LSTM": ([26.0,27.5,34.5],[77.0,70.3,64.7],[20.6,25.9,47.4]),
    "AE": ([21.5,31.4,38.1],[67.2,80.1,83.4],[23.1,24.3,26.1]),
    "TGCN": ([49.6,59.3,59.8],[79.3,79.6,80.4],[51.8,53.4,54.8]),
    "STGT": ([52.3,59.4,76.3],[81.5,83.7,84.8],[54.2,55.6,74.7]),
    "TL-STGT": ([54.8,65.1,79.7],[82.6,86.1,87.1],[53.6,59.6,76.8]),
}


def resid_forecaster(m, ds, idx, a):
    return ds["target_obs"][idx] - models.predict(m, ds["hist"][idx], a)


def resid_ae(m, ds, idx):
    return models.predict(m, ds["target_obs"][idx]) - ds["target_obs"][idx]


def resid_trivial(kind, ds, idx):
    """Zero-parameter reference residuals.

    persist: predict the next reading = the last observed reading. The dumbest
             possible forecaster; nothing is learned or fitted.
    zscore:  no forecast at all -- the standardized reading itself is the
             'residual', i.e. flag readings far from the benign mean.
    """
    if kind == "persist":
        return ds["target_obs"][idx] - ds["hist"][idx][:, -1, :]
    return ds["target_obs"][idx]


def adjacency(A, idx, mode, seed):
    """Ablation of the graph itself (Tier 2).

    real:     the C-Town SCADA topology.
    shuffled: same topology, node labels permuted -- destroys the correspondence
              between graph position and sensor while preserving graph structure.
              If results are unchanged, the topology carries no information.
    identity: no edges at all -- the graph convolution degenerates to per-node.
    """
    sub = A[np.ix_(idx, idx)]
    if mode == "identity":
        sub = np.eye(len(idx))
    elif mode == "shuffled":
        p = np.random.default_rng(10_000 + seed).permutation(len(idx))
        sub = sub[np.ix_(p, p)]
    return models.normalized_adj(sub)


def _eval_at(r, ds, s_list, cfg):
    """Evaluate one trained model's residual function at each detection batch size."""
    va, te = ds["val"], ds["test"]
    # Axis 6 — population the Mahalanobis mean/covariance is fitted on.
    fit_idx = ds["train"] if cfg["errfit"] == "all" else ds["train_benign"]
    rb, rv, rt = r(fit_idx), r(va), r(te)
    return {s: detect.evaluate_forecaster(rb, rv, ds["labels"][va], rt, ds["labels"][te],
                                          val_clean=ds["win_clean"][va], batch_s=s,
                                          thr_mode=cfg["thresh"])
            for s in s_list}


def _train_view(ds, cfg):
    """Axis 1 — which samples/targets the forecaster is fitted on.

    balanced: the paper's literal 80/10/10 "equal samples per class in each set",
              so training targets include attack-corrupted readings.
    benign:   only windows free of attack readings (what the detector implies).
    """
    if cfg["train"] == "balanced":
        return ds["train"], ds["target_obs"]
    return ds["train_benign"], ds["target_true"]


def eval_deep(name, ds, a, s_list, cfg, epochs=None):
    tr, tgt = _train_view(ds, cfg)
    if name == "ae":
        m = models.train(models.build("ae", cfg["window"], ds["n_nodes"]), tgt[tr], tgt[tr])
        r = lambda idx: resid_ae(m, ds, idx)
    else:
        an = a if name in GRAPH else None
        mdl = models.build(name, cfg["window"], ds["n_nodes"], transformer=cfg["transformer"])
        m = models.train(mdl, ds["hist"][tr], tgt[tr], an,
                         **({"epochs": epochs} if epochs else {}))
        r = lambda idx: resid_forecaster(m, ds, idx, an)
    return _eval_at(r, ds, s_list, cfg)


def run_size(A, size, delta_scale, s_list, seed, cfg):
    idx = data.figure_nodes(size)
    ds = data.make_datasets(idx, seed=seed, delta_scale=delta_scale, block=cfg["block"],
                            window=cfg["window"], ablate=cfg["ablate"],
                            replay_dt=cfg["replay_dt"])
    a = adjacency(A, idx, cfg["adj"], seed)
    out = {s: {} for s in s_list}
    for k in TRIVIAL:                                   # zero-parameter reference points
        per_s = _eval_at(lambda i, kk=k: resid_trivial(kk, ds, i), ds, s_list, cfg)
        for s in s_list:
            out[s][NAME[k]] = per_s[s]
    # Axis 10: the paper never states how the shallow models decide. Two readings:
    # classify the observed reading, or classify a persistence-residual feature.
    if cfg["shallow"] == "residual":
        feat = lambda i: ds["target_obs"][i] - ds["hist"][i][:, -1, :]
    else:
        feat = lambda i: ds["target_obs"][i]
    for k in ("svm", "rf", "lgbm"):                     # shallow: S-independent
        clf = models.shallow_classifier(k)
        clf.fit(feat(ds["train"]), ds["labels"][ds["train"]])
        met = detect.evaluate_classifier(ds["labels"][ds["test"]],
                                         clf.predict(feat(ds["test"])))
        for s in s_list:
            out[s][NAME[k]] = met
    for k in FORE:
        per_s = eval_deep(k, ds, a, s_list, cfg)
        for s in s_list:
            out[s][NAME[k]] = per_s[s]
    if cfg.get("tl_budget_match"):
        per_s = eval_deep("stgt", ds, a, s_list, cfg, epochs=models.EPOCHS * len(SIZES))
        for s in s_list:
            out[s][BUDGET_MATCH_NAME] = per_s[s]
    return out


def run_tl(A, delta_scale, s_list, seed, cfg):
    res, prev = {s: {} for s in s_list}, None
    for size in SIZES:
        idx = data.figure_nodes(size)
        ds = data.make_datasets(idx, seed=seed, delta_scale=delta_scale, block=cfg["block"],
                            window=cfg["window"], ablate=cfg["ablate"],
                            replay_dt=cfg["replay_dt"])
        a = adjacency(A, idx, cfg["adj"], seed)
        m = models.build("stgt", cfg["window"], size, transformer=cfg["transformer"])
        if prev is not None:
            m.load_state_dict(prev.state_dict())
            for grp in m.backbone():
                for p in grp.parameters():
                    p.requires_grad = False
        tr, tgt = _train_view(ds, cfg)
        m = models.train(m, ds["hist"][tr], tgt[tr], a)
        per_s = _eval_at(lambda i: resid_forecaster(m, ds, i, a), ds, s_list, cfg)
        for s in s_list:
            res[s][size] = per_s[s]
        for p in m.parameters():
            p.requires_grad = True
        prev = m
    return res


def report(per_size, tl, tag, s, fh):
    def w(line=""):
        print(line); fh.write(line + "\n")
    w(f"\n############ {tag}  S={s} ############")
    for metric in ("F1", "ACC", "DR"):
        w(f"\n================  {metric}  (ours | reported)  ================")
        w(f"{'model':9} " + " ".join(f"{z:>17}" for z in SIZES))
        for name in ORDER:
            cells = []
            for si, size in enumerate(SIZES):
                ours = (tl[size] if name == "TL-STGT" else per_size[size][name])[metric]
                if name in REPORTED:
                    rep = REPORTED[name][("F1","ACC","DR").index(metric)][si]
                    cells.append(f"{ours:6.1f} | {rep:5.1f}")
                else:                       # not in the paper: zero-parameter reference
                    cells.append(f"{ours:6.1f} |   -- ")
            w(f"{name:9} " + " ".join(f"{c:>17}" for c in cells))
    w(f"\n---- realized false-alarm rate (ours; the paper reports none) ----")
    w(f"{'model':9} " + " ".join(f"{z:>17}" for z in SIZES))
    for name in ORDER:
        cells = [f"{(tl[size] if name=='TL-STGT' else per_size[size][name])['FA']:6.1f}"
                 for size in SIZES]
        w(f"{name:9} " + " ".join(f"{c:>17}" for c in cells))
    w("\n* ZSCORE/PERSIST are zero-parameter reference points, not from the paper.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-scale", default="raw", choices=list(data.DELTA_SCALES))
    ap.add_argument("--batch-s", type=int, nargs="+", default=[detect.BATCH_S])
    ap.add_argument("--seed", type=int, default=0)
    # Interpretation axes (see AMBIGUITY_REGISTER.md). Defaults are the literal reading.
    ap.add_argument("--train", default="balanced", choices=["balanced", "benign"])
    ap.add_argument("--thresh", default="fa5", choices=["fa5", "maxf1"])
    ap.add_argument("--transformer", default="nodes", choices=["nodes", "time"])
    ap.add_argument("--errfit", default="normal", choices=["normal", "all"])
    ap.add_argument("--adj", default="real", choices=["real", "shuffled", "identity"])
    ap.add_argument("--window", type=int, default=data.WINDOW)
    ap.add_argument("--ablate", default="none", choices=["none", "space", "time"])
    ap.add_argument("--block", type=int, default=60)
    ap.add_argument("--replay-dt", default="random")
    ap.add_argument("--shallow", default="reading", choices=["reading", "residual"])
    ap.add_argument("--tl-budget-match", action="store_true",
                    help="also train scratch STGT for 3x epochs, matching the TL chain total")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = dict(train=args.train, thresh=args.thresh, transformer=args.transformer,
               errfit=args.errfit, adj=args.adj, window=args.window,
               ablate=args.ablate, block=args.block, replay_dt=args.replay_dt,
               shallow=args.shallow, tl_budget_match=args.tl_budget_match)
    tag = (f"delta={args.delta_scale} train={args.train} thr={args.thresh} "
           f"tf={args.transformer} fit={args.errfit} adj={args.adj} "
           f"W={args.window} ablate={args.ablate} block={args.block} "
           f"dt={args.replay_dt} shallow={args.shallow} seed={args.seed}")

    models.torch.manual_seed(args.seed)      # model init/shuffling; data seed passed separately
    A = data.graph()
    s_list = sorted(set(args.batch_s))
    per_size = {}
    for size in SIZES:
        print(f"[{tag} size {size}]", file=sys.stderr)
        per_size[size] = run_size(A, size, args.delta_scale, s_list, args.seed, cfg)
    print(f"[{tag} TL-STGT]", file=sys.stderr)
    tl = run_tl(A, args.delta_scale, s_list, args.seed, cfg)

    out = args.out or (f"results_{args.delta_scale}_{args.train}_{args.thresh}_"
                       f"{args.transformer}_{args.errfit}_seed{args.seed}.txt")
    with open(out, "w") as fh:
        for s in s_list:
            report({sz: per_size[sz][s] for sz in SIZES}, tl[s], tag, s, fh)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
