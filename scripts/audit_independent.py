import os, sys, json, pickle, hashlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, average_precision_score

from src.orca.models.orca_model import OrcaModel
from src.orca.preprocessing.normalizer import SERVICES, SERVICE2IDX, IDX2SERVICE

# 1. Checkpoint Verification
ckpt_path = Path("artifacts/models/orca_dev_checkpoint.pt")
h = hashlib.sha256()
with open(ckpt_path, "rb") as f:
    while chunk := f.read(8192): h.update(chunk)
ckpt_hash = h.hexdigest()
print(f"Model Checkpoint: {ckpt_path}")
print(f"SHA256 Hash: {ckpt_hash}")

device = torch.device("cpu")
model = OrcaModel(
    vocab_size=500, log_embed_dim=32, log_hidden_dim=64, log_num_layers=1,
    in_metrics=70, metric_hidden_dim=64, metric_num_layers=2,
    node_feat_dim=27, graph_hidden_dim=32, graph_num_layers=2, graph_heads=2,
    fusion_type="cross_attention", hidden_dim=64, num_services=10,
).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 2. Dataset Split Verification
proc_dir = Path("data/processed")
with open(proc_dir / "val_samples.pkl", "rb") as f: val_samples = pickle.load(f)
with open(proc_dir / "test_samples.pkl", "rb") as f: test_samples = pickle.load(f)

def run_forward(samples):
    all_logits = []
    all_anom_probs = []
    all_y_anom = []
    all_y_rc = []
    case_names = []
    with torch.no_grad():
        for s in samples:
            tb = {
                "metric_seq": torch.tensor(s["metric_seq"], dtype=torch.float32).unsqueeze(0),
                "metric_flat": torch.tensor(s["metric_flat"], dtype=torch.float32).unsqueeze(0),
                "log_tokens": torch.tensor(s["log_tokens"], dtype=torch.long).unsqueeze(0),
                "log_service_stats": torch.tensor(s["log_service_stats"], dtype=torch.float32).unsqueeze(0),
                "graph_nodes": torch.tensor(s["graph_nodes"], dtype=torch.float32).unsqueeze(0),
                "graph_adj": torch.tensor(s["graph_adj"], dtype=torch.float32).unsqueeze(0),
                "trace_stats": torch.tensor(s["trace_stats"], dtype=torch.float32).unsqueeze(0),
            }
            out = model(tb)
            all_logits.append(out["root_cause_logits"][0].numpy())
            all_anom_probs.append(out["anomaly_probs"][0, 1].item())
            all_y_anom.append(s["y_anomaly"])
            all_y_rc.append(s["y_root_cause"])
            case_names.append(s["case_name"])
    return np.array(all_logits), np.array(all_anom_probs), np.array(all_y_anom), np.array(all_y_rc), case_names

print("Running validation inference...")
val_logits, val_anom_probs, val_y_anom, val_y_rc, val_cases = run_forward(val_samples)
print("Running test inference...")
test_logits, test_anom_probs, test_y_anom, test_y_rc, test_cases = run_forward(test_samples)

print(f"Val samples: {len(val_samples)}, Test samples: {len(test_samples)}")
val_anom_mask = (val_y_anom == 1)
test_anom_mask = (test_y_anom == 1)
print(f"Val active anomaly windows: {val_anom_mask.sum()}")
print(f"Test active anomaly windows: {test_anom_mask.sum()}")

# 3. Fit Temperature on Validation Only (strictly active anomaly windows)
v_logits = val_logits[val_anom_mask]
v_targets = val_y_rc[val_anom_mask]

def nll_obj(T):
    scaled = v_logits / T
    max_l = np.max(scaled, axis=1, keepdims=True)
    lse = max_l + np.log(np.sum(np.exp(scaled - max_l), axis=1, keepdims=True))
    lp = scaled - lse
    return float(-np.mean(lp[np.arange(len(v_targets)), v_targets]))

opt_res = minimize_scalar(nll_obj, bounds=(0.05, 5.0), method="bounded")
T_opt = float(opt_res.x)
print(f"Fitted T on Validation Logits: {T_opt:.8f}")

with open("artifacts/models/calibration_params.json", "r") as f:
    saved_params = json.load(f)
saved_T = saved_params["temperature"]
print(f"Saved T in calibration_params.json: {saved_T:.8f}")
print(f"Exact Match: {abs(T_opt - saved_T) < 1e-6}")

# 4. Helper for softmax and calibration metrics
def softmax(z):
    max_z = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - max_z)
    return e / np.sum(e, axis=-1, keepdims=True)

def calc_calib_metrics(probs, targets, n_bins=10):
    N, C = probs.shape
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    bin_details = []

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        in_bin = (confidences >= b_low) & (confidences <= b_high) if i == 0 else (confidences > b_low) & (confidences <= b_high)
        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            diff = float(np.abs(avg_conf - avg_acc))
            weight = float(count / N)
            ece += weight * diff
            mce = max(mce, diff)
            bin_details.append({
                "bin": f"({b_low:.1f}, {b_high:.1f}]" if i > 0 else f"[{b_low:.1f}, {b_high:.1f}]",
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": avg_acc,
                "diff": diff
            })

    eps = 1e-12
    correct_probs = probs[np.arange(N), targets]
    nll = float(-np.mean(np.log(np.clip(correct_probs, eps, 1.0))))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), targets] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {"ece": ece, "mce": mce, "nll": nll, "brier": brier, "bin_details": bin_details}

# 5. Independent Metrics Computation on Test Split
t_logits = test_logits[test_anom_mask]
t_targets = test_y_rc[test_anom_mask]
t_cases = [test_cases[i] for i in range(len(test_cases)) if test_anom_mask[i]]

# Raw & Calibrated Probs
t_raw_probs = softmax(t_logits)
t_cal_probs = softmax(t_logits / T_opt)

# Monotonicity check
raw_rankings = np.argsort(-t_raw_probs, axis=1)
cal_rankings = np.argsort(-t_cal_probs, axis=1)
ranking_identical = np.array_equal(raw_rankings, cal_rankings)
print(f"Ranking Invariance (Raw == Calibrated): {ranking_identical}")

# RCA Metrics
def rca_eval(probs, targets):
    top1 = np.mean(np.argmax(probs, axis=1) == targets)
    top3 = np.mean([targets[i] in np.argsort(-probs[i])[:3] for i in range(len(targets))])
    top5 = np.mean([targets[i] in np.argsort(-probs[i])[:5] for i in range(len(targets))])
    ranks = [np.where(np.argsort(-probs[i]) == targets[i])[0][0] + 1 for i in range(len(targets))]
    mrr = np.mean([1.0 / r for r in ranks])
    correct_count = np.sum(np.argmax(probs, axis=1) == targets)
    return {"top1": top1, "top3": top3, "top5": top5, "mrr": mrr, "correct": int(correct_count), "total": len(targets)}

rca_active = rca_eval(t_cal_probs, t_targets)
rca_all = rca_eval(softmax(test_logits / T_opt), test_y_rc)

print("\n=== RCA PERFORMANCE (ACTIVE ANOMALY WINDOWS, N=144) ===")
print(f"Top-1 Accuracy: {rca_active['top1']*100:.2f}% ({rca_active['correct']}/{rca_active['total']})")
print(f"Top-3 Accuracy: {rca_active['top3']*100:.2f}%")
print(f"Top-5 Accuracy: {rca_active['top5']*100:.2f}%")
print(f"MRR:            {rca_active['mrr']:.4f}")

print("\n=== RCA PERFORMANCE (ALL WINDOWS, N=288) ===")
print(f"Top-1 Accuracy: {rca_all['top1']*100:.2f}% ({rca_all['correct']}/{rca_all['total']})")
print(f"Top-3 Accuracy: {rca_all['top3']*100:.2f}%")
print(f"MRR:            {rca_all['mrr']:.4f}")

# Anomaly Detection Metrics
pred_anom_labels = (test_anom_probs >= 0.5).astype(int)
cm = confusion_matrix(test_y_anom, pred_anom_labels)
tn, fp, fn, tp = cm.ravel()
p, r, f1, _ = precision_recall_fscore_support(test_y_anom, pred_anom_labels, average="binary")
roc_auc = roc_auc_score(test_y_anom, test_anom_probs)
pr_auc = average_precision_score(test_y_anom, test_anom_probs)

print("\n=== ANOMALY DETECTION PERFORMANCE (TEST SET, N=288) ===")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

# Calibration Metrics
cal_raw = calc_calib_metrics(t_raw_probs, t_targets)
cal_opt = calc_calib_metrics(t_cal_probs, t_targets)

print("\n=== CALIBRATION METRICS (ACTIVE ANOMALY WINDOWS, N=144) ===")
print(f"Uncalibrated ECE:   {cal_raw['ece']*100:.2f}% | Calibrated ECE:   {cal_opt['ece']*100:.2f}%")
print(f"Uncalibrated MCE:   {cal_raw['mce']*100:.2f}% | Calibrated MCE:   {cal_opt['mce']*100:.2f}%")
print(f"Uncalibrated NLL:   {cal_raw['nll']:.4f}    | Calibrated NLL:   {cal_opt['nll']:.4f}")
print(f"Uncalibrated Brier: {cal_raw['brier']:.4f}  | Calibrated Brier: {cal_opt['brier']:.4f}")

# Per-Scenario Breakdown
print("\n=== PER-SCENARIO PERFORMANCE BREAKDOWN ===")
for c_name in sorted(list(set(t_cases))):
    idx = [i for i, c in enumerate(t_cases) if c == c_name]
    sub_probs = t_cal_probs[idx]
    sub_raw_probs = t_raw_probs[idx]
    sub_targets = t_targets[idx]
    sub_res = rca_eval(sub_probs, sub_targets)
    avg_cal_conf = np.mean(np.max(sub_probs, axis=1))
    avg_raw_conf = np.mean(np.max(sub_raw_probs, axis=1))
    gt_svc = IDX2SERVICE[sub_targets[0]]
    pred_svc_counts = {}
    for p_idx in np.argmax(sub_probs, axis=1):
        s_name = IDX2SERVICE[p_idx]
        pred_svc_counts[s_name] = pred_svc_counts.get(s_name, 0) + 1
    print(f"Scenario: {c_name}")
    print(f"  Ground Truth: {gt_svc} | Evaluated Windows: {len(idx)}")
    print(f"  Top-1 Accuracy: {sub_res['top1']*100:.2f}% ({sub_res['correct']}/{sub_res['total']})")
    print(f"  Avg Raw Confidence: {avg_raw_conf*100:.2f}% | Avg Calibrated Confidence: {avg_cal_conf*100:.2f}%")
    print(f"  Predicted Distribution: {pred_svc_counts}")

# Per-Service Breakdown
print("\n=== PER-SERVICE PERFORMANCE BREAKDOWN ===")
present_services = sorted(list(set(t_targets)))
for s_idx in present_services:
    s_name = IDX2SERVICE[s_idx]
    mask = (t_targets == s_idx)
    sub_probs = t_cal_probs[mask]
    sub_targets = t_targets[mask]
    sub_res = rca_eval(sub_probs, sub_targets)
    print(f"Service: {s_name} (idx={s_idx})")
    print(f"  Support: {len(sub_targets)} | Accuracy: {sub_res['top1']*100:.2f}% ({sub_res['correct']}/{sub_res['total']})")

# Binned Reliability Table
print("\n=== TEST RELIABILITY DIAGRAM DATA (CALIBRATED) ===")
for b in cal_opt['bin_details']:
    print(f"Bin {b['bin']:>12}: Count={b['count']:>2}, Avg Conf={b['avg_confidence']*100:>5.1f}%, Acc={b['accuracy']*100:>5.1f}%, Error={b['diff']*100:>5.1f}%")
