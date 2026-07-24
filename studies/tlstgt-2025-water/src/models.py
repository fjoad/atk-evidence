"""Paper-literal models for the TL-STGT paper (Section III, V-B).

Shallow (SVM/RF/LGBM): supervised classifiers on the observed reading vector.
Forecasters (FFNN 5x500 tanh; LSTM 3x100 tanh; TGCN 3-GCN+GRU 64u; STGT
GCN->GRU->maxpool->Dense128 ReLU->Transformer 8h/2blocks->head): predict the next
reading from the benign history window; detection = Mahalanobis on residual
(observed - predicted).  AE: 4-layer ReLU reconstruction autoencoder on the reading
vector; detection = Mahalanobis on reconstruction error.
TL-STGT: STGT + Algorithm-1 transfer (warm-start, freeze GCN+GRU, fine-tune rest),
progressively 10->20->31.  GCN follows eq5 (tanh, neighbour sum, no self-loop).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

EPOCHS, BATCH, LR = 30, 32, 1e-3
torch.manual_seed(0)


def shallow_classifier(kind: str):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    if kind == "svm":
        return SVC(gamma=0.07, C=10)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    if kind == "lgbm":
        return LGBMClassifier(learning_rate=0.5, max_depth=2, n_estimators=100,
                              random_state=0, verbose=-1, n_jobs=1)
    raise ValueError(kind)


class FFNN(nn.Module):                       # 5 layers, 500, tanh; window -> next reading
    def __init__(self, w, n):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
            nn.Linear(w * n, 500), nn.Tanh(), nn.Linear(500, 500), nn.Tanh(),
            nn.Linear(500, 500), nn.Tanh(), nn.Linear(500, 500), nn.Tanh(), nn.Linear(500, n))
    def forward(self, x): return self.net(x)


class LSTMNet(nn.Module):                     # 3 layers, 100, tanh
    def __init__(self, w, n):
        super().__init__()
        self.lstm = nn.LSTM(n, 100, num_layers=3, batch_first=True)
        self.head = nn.Linear(100, n)
    def forward(self, x):
        o, _ = self.lstm(x); return self.head(torch.tanh(o[:, -1]))


class AENet(nn.Module):                       # 4-layer ReLU reconstruction AE on a reading
    def __init__(self, w, n):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, n))
    def forward(self, x): return self.dec(self.enc(x))


class GraphConv(nn.Module):                   # eq5: tanh( Ahat (X W) )
    def __init__(self, cin, cout):
        super().__init__(); self.lin = nn.Linear(cin, cout)
    def forward(self, x, a): return torch.tanh(a @ self.lin(x))


def _core(x, gcns, gru, a):
    B, W, N = x.shape
    steps = []
    for t in range(W):
        z = x[:, t].unsqueeze(-1)
        for g in gcns:
            z = g(z, a)
        steps.append(z)
    h = torch.stack(steps, 1).permute(0, 2, 1, 3).reshape(B * N, W, -1)
    o, _ = gru(h)
    return o.max(dim=1).values.reshape(B, N, -1)          # temporal max-pool -> (B,N,H)


class TGCN(nn.Module):                        # 3 graph-conv layers, 64 units, GRU
    def __init__(self, w, n, hidden=64):
        super().__init__()
        self.gcns = nn.ModuleList([GraphConv(1, hidden), GraphConv(hidden, hidden), GraphConv(hidden, hidden)])
        self.gru = nn.GRU(hidden, hidden, batch_first=True); self.head = nn.Linear(hidden, 1)
    def forward(self, x, a): return self.head(_core(x, self.gcns, self.gru, a)).squeeze(-1)


class STGT(nn.Module):                        # GCN->GRU->maxpool->Dense128 ReLU->Transformer->head
    def __init__(self, w, n, hidden=64, heads=8, dmodel=128):
        super().__init__()
        self.gcns = nn.ModuleList([GraphConv(1, hidden)])
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(hidden, dmodel), nn.ReLU())
        enc = nn.TransformerEncoderLayer(dmodel, heads, dim_feedforward=dmodel, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=2); self.head = nn.Linear(dmodel, 1)
    def backbone(self): return [self.gcns, self.gru]
    def forward(self, x, a):
        s = self.dense(_core(x, self.gcns, self.gru, a)); s = self.tr(s)
        return self.head(s).squeeze(-1)


class STGTTime(nn.Module):     # ALT reading: transformer OVER TIME (before pooling)
    def __init__(self, w, n, hidden=64, heads=8, dmodel=128):
        super().__init__()
        self.gcns = nn.ModuleList([GraphConv(1, hidden)])
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(hidden, dmodel), nn.ReLU())
        enc = nn.TransformerEncoderLayer(dmodel, heads, dim_feedforward=dmodel, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=2); self.head = nn.Linear(dmodel, 1)
    def backbone(self): return [self.gcns, self.gru]
    def forward(self, x, a):
        B, W, N = x.shape
        steps = []
        for t in range(W):
            z = x[:, t].unsqueeze(-1)
            for g in self.gcns:
                z = g(z, a)
            steps.append(z)
        h = torch.stack(steps, 1).permute(0, 2, 1, 3).reshape(B * N, W, -1)
        o, _ = self.gru(h)                     # (B*N, W, H) full sequence
        s = self.tr(self.dense(o))[:, -1]      # transformer over TIME, take last step
        return self.head(s).reshape(B, N)


def normalized_adj(A: np.ndarray) -> torch.Tensor:        # D^-1/2 A D^-1/2, no self-loop (eq5)
    d = A.sum(1); d[d == 0] = 1.0
    dinv = np.diag(1.0 / np.sqrt(d))
    return torch.tensor(dinv @ A @ dinv, dtype=torch.float32)


def train(model, X, Y, a=None, epochs=EPOCHS):
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR)
    loss_fn = nn.MSELoss(); X, Y = torch.tensor(X), torch.tensor(Y); model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            b = perm[i:i + BATCH]; opt.zero_grad()
            out = model(X[b], a) if a is not None else model(X[b])
            loss_fn(out, Y[b]).backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, X, a=None):
    model.eval(); X = torch.tensor(X)
    return (model(X, a) if a is not None else model(X)).numpy()


def build(name, w, n, transformer="nodes"):
    """Ambiguity Axis 5: `transformer="time"` selects the STGTTime reading, in which
    the transformer attends over time before pooling (as the text describes) rather
    than over nodes after pooling (as Fig. 2 draws)."""
    table = {"ffnn": FFNN, "lstm": LSTMNet, "ae": AENet, "tgcn": TGCN, "stgt": STGT}
    if name == "stgt" and transformer == "time":
        return STGTTime(w, n)
    return table[name](w, n)
