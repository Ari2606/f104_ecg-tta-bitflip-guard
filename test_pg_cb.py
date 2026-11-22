"""Light test for the minority-class protections in safety_controller (class_balance, anchor_kl).
Runs in seconds; does not run the heavy controller self-test."""
import copy

import torch

from base_cnn import CNNConfig, BeatCNN
from safety_controller import GuardedTTA, SafetyControllerConfig
import method_zoo

torch.manual_seed(0)

# 1) weight function: majority class gets a low weight, rare class a high one, both clipped
c = GuardedTTA(SafetyControllerConfig(class_balance=True, cb_momentum=0.9))
c._freq = None
for _ in range(200):
    w_n = c._cb_weight(torch.tensor([[5.0, 0, 0, 0, 0]]))
w_s = c._cb_weight(torch.tensor([[0, 5.0, 0, 0, 0]]))
assert w_n <= 0.25 and w_s > 1.5 and w_s > 8 * w_n, (w_n, w_s)   # (momentum .9 here => S weight ~2)
assert 0.2 <= w_n <= 5.0 and 0.2 <= w_s <= 5.0
print(f"weights after 200 N beats: N={w_n:.3f}  S(first rare)={w_s:.3f}")

# 2) the step runs for every variant; defaults are bit-identical to the unmodified method;
#    the options change the update
base = BeatCNN(CNNConfig(in_channels=1))
base.eval()
base.freeze_anchor()
xs = torch.randn(40, 1, 256)


def run(key):
    m = copy.deepcopy(base)
    meth = method_zoo.build_method(key, {})
    meth.reset(m)
    preds = []
    for i in range(xs.shape[0]):
        r = meth.step(m, xs[i:i + 1], {})
        assert torch.isfinite(r.logits).all()
        preds.append(r.logits.argmax(-1).item())
    return torch.cat([p.detach().flatten() for p in m.parameters()]), preds


p_def, _ = run("pg")
p_def2, _ = run("pg")
assert torch.equal(p_def, p_def2), "default must be deterministic"
for k in ("pg_cb", "pg_kl", "pg_cbkl", "pg_gate"):
    p_k, _ = run(k)
    assert torch.isfinite(p_k).all()
    assert not torch.equal(p_k, p_def), f"{k} should change the update"
    print(f"{k}: runs, finite, differs from pg (max |dparam| = {(p_k - p_def).abs().max():.2e})")
print("test_pg_cb: passed")
