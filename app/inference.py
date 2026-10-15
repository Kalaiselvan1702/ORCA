"""Model inference runner for ORCA serving layer."""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

from src.orca.models.orca_model import OrcaModel
from src.orca.preprocessing.normalizer import SERVICES, SERVICE2IDX, CANONICAL_METRICS, normalize_service_name
from src.orca.features.log_features import ERROR_KEYWORDS
from src.orca.evaluation.calibration import TemperatureCalibrator, assess_certainty_status
from src.orca.utils.logger import get_logger

logger = get_logger("orca.app.inference")


class OrcaInferenceEngine:
    """Manages model loading and inference for incident payloads."""

    def __init__(
        self,
        checkpoint_path: str = "artifacts/models/orca_dev_checkpoint.pt",
        calibration_path: str = "artifacts/models/calibration_params.json",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path)
        self.calibration_path = Path(calibration_path)
        self.calibrator = TemperatureCalibrator.load(self.calibration_path)
        self.is_trained = False
        self.model = OrcaModel(
            vocab_size=500, log_embed_dim=32, log_hidden_dim=64, log_num_layers=1,
            in_metrics=70, metric_hidden_dim=64, metric_num_layers=2,
            node_feat_dim=27, graph_hidden_dim=32, graph_num_layers=2, graph_heads=2,
            fusion_type="cross_attention", hidden_dim=64, num_services=10,
        ).to(self.device)

        if self.checkpoint_path.exists():
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.is_trained = True
                logger.info(f"Successfully loaded trained ORCA checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {self.checkpoint_path}: {e}")
        else:
            logger.warning(f"No checkpoint found at {self.checkpoint_path}. Using initialization weights.")

        self.model.eval()

    def build_tensors_from_payload(
        self, payload: Dict[str, Any], window_len: int = 15
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Convert API payload into model tensors along with telemetry coverage metadata."""
        num_services = len(SERVICES)

        # 1. Metrics: (1, W, 70)
        metric_seq = np.zeros((1, window_len, len(CANONICAL_METRICS)), dtype=np.float32)
        raw_metrics = payload.get("metrics") or {}
        metric_keys_present = []
        for c_idx, col in enumerate(CANONICAL_METRICS):
            if col in raw_metrics and raw_metrics[col]:
                vals = raw_metrics[col][:window_len]
                metric_seq[0, :len(vals), c_idx] = vals
                metric_keys_present.append(col)
        metric_flat = np.zeros((1, 280), dtype=np.float32)

        # 2. Logs: tokens (1, 64) and service stats (1, 20)
        log_tokens = np.zeros((1, 64), dtype=np.int64)
        log_tokens[0, 0] = 2  # <CLS>
        log_service_stats = np.zeros((1, num_services * 2), dtype=np.float32)

        raw_logs = payload.get("logs") or []
        for l in raw_logs:
            svc = normalize_service_name(l.get("service"))
            msg = str(l.get("message", "")).lower()
            if svc in SERVICE2IDX:
                s_idx = SERVICE2IDX[svc]
                log_service_stats[0, s_idx] += 1.0
                if any(k in msg for k in ERROR_KEYWORDS):
                    log_service_stats[0, num_services + s_idx] += 1.0

        # 3. Traces: graph_nodes (1, 10, 4), graph_adj (1, 10, 10), trace_stats (1, 4)
        # 3. Traces and Multimodal Node features (1, 10, 27)
        graph_nodes = np.zeros((1, num_services, 27), dtype=np.float32)
        graph_adj = np.eye(num_services, dtype=np.float32)[np.newaxis, :, :]
        trace_stats = np.zeros((1, 4), dtype=np.float32)

        raw_traces = payload.get("traces") or []
        span_to_svc = {}
        durations = []
        for tr in raw_traces:
            svc = normalize_service_name(tr.get("service"))
            s_id = tr.get("span_id")
            dur = float(tr.get("duration_ms", 0.0))
            code = int(tr.get("status_code", 0))
            durations.append(dur)
            if svc in SERVICE2IDX and s_id:
                s_idx = SERVICE2IDX[svc]
                span_to_svc[s_id] = s_idx
                graph_nodes[0, s_idx, 0] += 1.0
                graph_nodes[0, s_idx, 1] += dur
                if dur > graph_nodes[0, s_idx, 2]:
                    graph_nodes[0, s_idx, 2] = dur
                if code != 0:
                    graph_nodes[0, s_idx, 3] += 1.0

        # Normalize duration and counts in graph_nodes
        for s_idx in range(num_services):
            cnt = graph_nodes[0, s_idx, 0]
            if cnt > 0:
                graph_nodes[0, s_idx, 1] /= cnt
                graph_nodes[0, s_idx, 3] /= cnt
            graph_nodes[0, s_idx, 0] = np.log1p(graph_nodes[0, s_idx, 0])
            graph_nodes[0, s_idx, 1] = np.log1p(graph_nodes[0, s_idx, 1])
            graph_nodes[0, s_idx, 2] = np.log1p(graph_nodes[0, s_idx, 2])

            # Multimodal enrich: logs and metrics
            l_cnt = log_service_stats[0, s_idx]
            l_err = log_service_stats[0, num_services + s_idx]
            graph_nodes[0, s_idx, 4] = np.log1p(l_cnt)
            graph_nodes[0, s_idx, 5] = np.log1p(l_err)

            s_m = metric_seq[0, :, s_idx * 7 : (s_idx + 1) * 7]
            graph_nodes[0, s_idx, 6:13] = np.mean(s_m, axis=0)
            graph_nodes[0, s_idx, 13:20] = np.max(s_m, axis=0)
            graph_nodes[0, s_idx, 20:27] = np.std(s_m, axis=0)

        # Construct adjacency links
        for tr in raw_traces:
            p_id = tr.get("parent_span_id")
            c_id = tr.get("span_id")
            if p_id in span_to_svc and c_id in span_to_svc:
                p_idx = span_to_svc[p_id]
                c_idx = span_to_svc[c_id]
                graph_adj[0, p_idx, c_idx] += 1.0

        # Row-normalize adj
        row_sum = graph_adj.sum(axis=-1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        graph_adj = graph_adj / row_sum

        trace_stats[0, 0] = len(raw_traces)
        trace_stats[0, 1] = sum(1 for tr in raw_traces if tr.get("status_code", 0) != 0)
        trace_stats[0, 2] = float(np.mean(durations)) if durations else 0.0
        trace_stats[0, 3] = float(np.max(durations)) if durations else 0.0

        batch = {
            "metric_seq": torch.tensor(metric_seq, dtype=torch.float32, device=self.device),
            "metric_flat": torch.tensor(metric_flat, dtype=torch.float32, device=self.device),
            "log_tokens": torch.tensor(log_tokens, dtype=torch.long, device=self.device),
            "log_service_stats": torch.tensor(log_service_stats, dtype=torch.float32, device=self.device),
            "graph_nodes": torch.tensor(graph_nodes, dtype=torch.float32, device=self.device),
            "graph_adj": torch.tensor(graph_adj, dtype=torch.float32, device=self.device),
            "trace_stats": torch.tensor(trace_stats, dtype=torch.float32, device=self.device),
        }

        # Telemetry metadata
        metrics_available = len(metric_keys_present) > 0
        logs_available = len(raw_logs) > 0
        traces_available = len(raw_traces) > 0
        is_empty = (not metrics_available and not logs_available and not traces_available)

        if is_empty:
            analysis_mode = "INSUFFICIENT_DATA"
        elif metrics_available and logs_available and traces_available:
            analysis_mode = "COMPLETE"
        else:
            analysis_mode = "PARTIAL"

        telemetry_meta = {
            "analysis_mode": analysis_mode,
            "logs_available": logs_available,
            "metrics_available": metrics_available,
            "traces_available": traces_available,
            "log_count": len(raw_logs),
            "metric_channels_present": len(metric_keys_present),
            "trace_span_count": len(raw_traces),
            "is_empty": is_empty,
        }

        return batch, telemetry_meta

    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        incident_id: str = "incident_unknown",
        telemetry_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform forward inference on a pre-constructed tensor batch."""
        if telemetry_meta is None:
            telemetry_meta = {
                "analysis_mode": "COMPLETE",
                "logs_available": True,
                "metrics_available": True,
                "traces_available": True,
                "log_count": 0,
                "metric_channels_present": len(CANONICAL_METRICS),
                "trace_span_count": 0,
                "is_empty": False,
            }

        # Handle complete missing telemetry truthfully
        if telemetry_meta.get("is_empty"):
            return {
                "incident_id": incident_id,
                "anomaly": False,
                "anomaly_score": 0.0,
                "severity": "LOW",
                "root_cause": {
                    "service": "UNKNOWN",
                    "confidence": 0.0,
                    "raw_confidence": 0.0,
                    "certainty_status": "UNCERTAIN",
                    "margin": 0.0,
                },
                "ranked_root_causes": [],
                "affected_services": [],
                "certainty_status": "UNCERTAIN",
                "telemetry_coverage": telemetry_meta,
                "modality_contributions": {
                    "logs": 0.0,
                    "metrics": 0.0,
                    "traces": 0.0,
                },
                "is_trained_model": self.is_trained,
            }

        logs_avail = bool(telemetry_meta.get("logs_available", True))
        metrics_avail = bool(telemetry_meta.get("metrics_available", True))
        traces_avail = bool(telemetry_meta.get("traces_available", True))

        with torch.no_grad():
            outputs = self.model(
                batch,
                ablate_logs=not logs_avail,
                ablate_metrics=not metrics_avail,
                ablate_traces=not traces_avail,
            )

        anom_prob = float(outputs["anomaly_probs"][0, 1].item())
        is_anom = bool(anom_prob >= 0.5)

        rc_logits = outputs["root_cause_logits"][0].cpu().numpy()
        raw_rc_probs = outputs["root_cause_probs"][0].cpu().numpy()
        cal_rc_probs = self.calibrator.calibrate_probs(rc_logits)

        ranked_indices = np.argsort(-cal_rc_probs)
        ranked_candidates = [
            {
                "service": SERVICES[idx],
                "score": float(cal_rc_probs[idx]),
                "raw_score": float(raw_rc_probs[idx]),
            }
            for idx in ranked_indices
        ]

        top_service = ranked_candidates[0]["service"]
        top_conf = ranked_candidates[0]["score"]
        raw_top_conf = ranked_candidates[0]["raw_score"]
        margin = float(ranked_candidates[0]["score"] - ranked_candidates[1]["score"]) if len(ranked_candidates) > 1 else 1.0

        certainty_status = assess_certainty_status(top_conf, margin, is_anom)

        # Detect non-root services manifesting cascading telemetry symptoms
        affected_services = []
        if "graph_nodes" in batch and traces_avail:
            gn = batch["graph_nodes"][0].cpu().numpy()
            for s_idx, svc_name in enumerate(SERVICES):
                if svc_name == top_service:
                    continue
                trace_dur = float(gn[s_idx, 1])
                lat50_dev = float(gn[s_idx, 8]) if gn.shape[-1] >= 13 else 0.0
                lat90_dev = float(gn[s_idx, 9]) if gn.shape[-1] >= 13 else 0.0
                err_rate = float(gn[s_idx, 3])

                if trace_dur > 2.0 or lat50_dev > 1.5 or lat90_dev > 1.5 or err_rate > 0.05:
                    symptom_parts = []
                    if trace_dur > 2.0:
                        symptom_parts.append("elevated RPC duration")
                    if lat50_dev > 1.5 or lat90_dev > 1.5:
                        symptom_parts.append(f"metric latency (+{max(lat50_dev, lat90_dev):.1f} std)")
                    if err_rate > 0.05:
                        symptom_parts.append(f"RPC failures ({err_rate*100:.1f}%)")

                    impact = max(trace_dur / 5.0, max(lat50_dev, lat90_dev) / 4.0)
                    affected_services.append({
                        "service": svc_name,
                        "symptom": ", ".join(symptom_parts) if symptom_parts else "cascading latency",
                        "impact_score": round(min(impact, 1.0), 3),
                    })

        # Severity
        sev_logits = outputs["severity_logits"][0].cpu().numpy()
        sev_classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        severity = sev_classes[int(np.argmax(sev_logits))]
        if anom_prob < 0.5:
            severity = "LOW"
        elif anom_prob > 0.85:
            severity = "HIGH"

        # Modality contribution estimates based on latent norm
        z_log_norm = float(torch.norm(outputs["z_log"]).item()) if logs_avail else 0.0
        z_met_norm = float(torch.norm(outputs["z_metric"]).item()) if metrics_avail else 0.0
        z_gra_norm = float(torch.norm(outputs["z_graph"]).item()) if traces_avail else 0.0
        total_norm = max(z_log_norm + z_met_norm + z_gra_norm, 1e-6)

        modality_contributions = {
            "logs": float(z_log_norm / total_norm),
            "metrics": float(z_met_norm / total_norm),
            "traces": float(z_gra_norm / total_norm),
        }

        return {
            "incident_id": incident_id,
            "anomaly": is_anom,
            "anomaly_score": round(anom_prob, 4),
            "severity": severity,
            "root_cause": {
                "service": top_service,
                "confidence": round(top_conf, 4),
                "raw_confidence": round(raw_top_conf, 4),
                "certainty_status": certainty_status,
                "margin": round(margin, 4),
            },
            "ranked_root_causes": ranked_candidates[:5],
            "affected_services": affected_services[:5],
            "certainty_status": certainty_status,
            "telemetry_coverage": telemetry_meta,
            "modality_contributions": modality_contributions,
            "is_trained_model": self.is_trained,
        }

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform full forward inference and return structured diagnosis."""
        batch, telemetry_meta = self.build_tensors_from_payload(payload)
        return self.predict_batch(
            batch,
            incident_id=payload.get("incident_id", "incident_unknown"),
            telemetry_meta=telemetry_meta,
        )

