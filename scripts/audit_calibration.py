"""Audit and benchmark calibration metrics on validation and test sets."""
import pickle
import torch
import numpy as np
from pathlib import Path
from src.orca.models.orca_model import OrcaModel
from src.orca.preprocessing.normalizer import SERVICES

def compute_calibration_metrics(probs, targets, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_details = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            abs_diff = float(np.abs(avg_confidence_in_bin - accuracy_in_bin))
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)
            bin_details.append({
                "bin": f"({bin_lower:.1f}, {bin_upper:.1f}]",
                "count": int(np.sum(in_bin)),
                "avg_conf": avg_confidence_in_bin,
                "accuracy": accuracy_in_bin,
                "error": abs_diff,
            })

    eps = 1e-12
    nll = float(-np.mean(np.log(probs[np.arange(len(targets)), targets] + eps)))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(targets)), targets] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "nll": nll,
        "brier": brier,
        "bin_details": bin_details,
    }

def main():
    ckpt = torch.load("artifacts/models/orca_dev_checkpoint.pt", map_location="cpu", weights_only=False)
    model = OrcaModel(
        vocab_size=500, log_embed_dim=32, log_hidden_dim=64, log_num_layers=1,
        in_metrics=70, metric_hidden_dim=64, metric_num_layers=2,
        node_feat_dim=27, graph_hidden_dim=32, graph_num_layers=2, graph_heads=2,
        fusion_type="cross_attention", hidden_dim=64, num_services=10
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for split_name in ["val", "test"]:
        with open(f"data/processed/{split_name}_samples.pkl", "rb") as f:
            samples = pickle.load(f)
        
        anom_samples = [s for s in samples if s["y_anomaly"] == 1]
        all_logits = []
        all_probs = []
        all_targets = []
        all_margins = []

        for s in anom_samples:
            tb = {
                "metric_seq": torch.tensor(s["metric_seq"]).unsqueeze(0),
                "log_tokens": torch.tensor(s["log_tokens"]).unsqueeze(0),
                "log_service_stats": torch.tensor(s["log_service_stats"]).unsqueeze(0),
                "graph_nodes": torch.tensor(s["graph_nodes"]).unsqueeze(0),
                "graph_adj": torch.tensor(s["graph_adj"]).unsqueeze(0),
                "trace_stats": torch.tensor(s["trace_stats"]).unsqueeze(0),
            }
            with torch.no_grad():
                out = model(tb)
            lg = out["root_cause_logits"][0].numpy()
            p = out["root_cause_probs"][0].numpy()
            order = np.argsort(-p)
            margin = p[order[0]] - p[order[1]]
            all_logits.append(lg)
            all_probs.append(p)
            all_targets.append(s["y_root_cause"])
            all_margins.append(margin)

        probs_arr = np.array(all_probs)
        targets_arr = np.array(all_targets)
        top1 = np.mean(np.argmax(probs_arr, axis=1) == targets_arr)

        cal = compute_calibration_metrics(probs_arr, targets_arr)
        print(f"=== Split: {split_name.upper()} ({len(anom_samples)} active incident windows) ===")
        print(f"  Top-1 Accuracy: {top1*100:.2f}%")
        print(f"  Mean Margin:    {np.mean(all_margins)*100:.2f}%")
        print(f"  ECE (10-bin):   {cal['ece']*100:.2f}%")
        print(f"  MCE:            {cal['mce']*100:.2f}%")
        print(f"  NLL:            {cal['nll']:.4f}")
        print(f"  Brier Score:    {cal['brier']:.4f}")
        print("  Bins:")
        for b in cal["bin_details"]:
            print(f"    {b['bin']}: count={b['count']}, avg_conf={b['avg_conf']*100:.1f}%, acc={b['accuracy']*100:.1f}%, diff={b['error']*100:.1f}%")
        print()

if __name__ == "__main__":
    main()
