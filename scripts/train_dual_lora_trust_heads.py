"""Fit, temperature-calibrate and freeze the two six-feature Trust heads."""

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "semantic_match", "provenance_freshness", "version_consistency",
    "conflict_degree", "support_coverage", "source_reliability",
)
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/dual_lora_trust_supervision")
parser.add_argument("--out", default="models/nb_recall_dual_lora_trust_heads")
args = parser.parse_args()


def read(name):
    return [json.loads(line) for line in (ROOT / args.data / f"{name}.jsonl").open()]


train = read("train")
calibration = read("calibration")
if not train or not calibration:
    raise RuntimeError("Both train and calibration supervision are required")


def matrix(rows):
    return np.asarray([[float(row["features"][name]) for name in FEATURE_NAMES]
                       for row in rows], dtype=np.float64)


X_train, X_cal = matrix(train), matrix(calibration)


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -50, 50)))


def ece(labels, probabilities, bins=10):
    result = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1],
                         np.linspace(0, 1, bins + 1)[1:]):
        selected = ((probabilities >= low) &
                    (probabilities < high if high < 1 else probabilities <= high))
        if selected.any():
            result += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(result)


def fit_head(label_name):
    train_idx = np.asarray([i for i, row in enumerate(train)
                            if row["labels"].get(label_name) is not None])
    cal_idx = np.asarray([i for i, row in enumerate(calibration)
                          if row["labels"].get(label_name) is not None])
    y_train = np.asarray([train[i]["labels"][label_name] for i in train_idx], dtype=int)
    y_cal = np.asarray([calibration[i]["labels"][label_name] for i in cal_idx], dtype=int)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_cal)) != 2:
        raise RuntimeError(f"{label_name} requires both classes in train and calibration")
    model = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=20261122))
    model.fit(X_train[train_idx], y_train)
    logits = model.decision_function(X_cal[cal_idx])

    def nll(log_temperature):
        probabilities = np.clip(sigmoid(logits / math.exp(log_temperature)), 1e-8, 1 - 1e-8)
        return float(-np.mean(y_cal * np.log(probabilities) +
                              (1 - y_cal) * np.log(1 - probabilities)))

    optimum = minimize_scalar(nll, bounds=(math.log(0.1), math.log(50)), method="bounded")
    temperature = math.exp(optimum.x)
    probabilities = sigmoid(logits / temperature)
    metrics = {
        "train_rows": len(train_idx), "calibration_rows": len(cal_idx),
        "train_positive_rate": float(y_train.mean()),
        "calibration_positive_rate": float(y_cal.mean()),
        "calibration_auroc": float(roc_auc_score(y_cal, probabilities)),
        "calibration_brier": float(brier_score_loss(y_cal, probabilities)),
        "calibration_ece_10bin": ece(y_cal, probabilities),
    }
    return model, temperature, probabilities, cal_idx, metrics


evidence, temp_e, p_e, idx_e, metrics_e = fit_head("evidence_supported")
fresh, temp_f, p_f, idx_f, metrics_f = fit_head("fresh_at_query")

# Threshold calibration uses rows with both labels defined.
joint = sorted(set(idx_e.tolist()) & set(idx_f.tolist()))
joint_position_e = {index: position for position, index in enumerate(idx_e.tolist())}
joint_position_f = {index: position for position, index in enumerate(idx_f.tolist())}
tau = np.asarray([p_e[joint_position_e[index]] * p_f[joint_position_f[index]] for index in joint])
correct = np.asarray([calibration[index]["labels"]["correct_at_query"] for index in joint], dtype=int)
target_risk, confidence = 0.10, 0.95

tau_hi = None
best_coverage = -1
for threshold in np.unique(tau):
    selected = tau >= threshold
    count = int(selected.sum())
    if count < 20:
        continue
    errors = int((1 - correct[selected]).sum())
    upper = float(beta.ppf(confidence, errors + 1, count - errors)) if errors < count else 1.0
    if upper <= target_risk and count > best_coverage:
        tau_hi = float(threshold)
        best_coverage = count

tau_lo = None
best_ignore = -1
for threshold in np.unique(tau):
    selected = tau < threshold
    count = int(selected.sum())
    if count < 20:
        continue
    successes = int(correct[selected].sum())
    upper_success = float(beta.ppf(confidence, successes + 1, count - successes)) if successes < count else 1.0
    if upper_success <= target_risk and count > best_ignore:
        tau_lo = float(threshold)
        best_ignore = count

if tau_hi is None:
    tau_hi = float(np.quantile(tau, 0.90))
    hi_source = "fallback_90th_quantile"
else:
    hi_source = "conformal_answer_risk"
if tau_lo is None:
    tau_lo = float(np.quantile(tau, 0.10))
    lo_source = "fallback_10th_quantile"
else:
    lo_source = "conformal_low_correctness"
if tau_lo >= tau_hi:
    tau_lo = max(0.0, tau_hi * 0.5)
    lo_source += "_adjusted_below_tau_hi"

middle = (tau >= tau_lo) & (tau < tau_hi)
middle_indices = [joint[i] for i in np.flatnonzero(middle)]
resolves = [calibration[index]["labels"].get("reobserve_resolves", 0)
            for index in middle_indices]
costs = [calibration[index]["audit_only"].get("next_observation_cost_steps")
         for index in middle_indices]
costs = [float(value) for value in costs if value is not None]
rho = float(np.mean(resolves)) if resolves else 0.01
c_obs = float(np.median(costs)) if costs else 1.0
rho = min(1.0, max(0.01, rho))

out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
joblib.dump(evidence, out / "tau_evidence.joblib")
joblib.dump(fresh, out / "tau_fresh.joblib")
report = {
    "feature_names": list(FEATURE_NAMES),
    "heads": {"tau_evidence": metrics_e, "tau_fresh": metrics_f},
    "temperature": {"tau_evidence": temp_e, "tau_fresh": temp_f},
    "tau_definition": "tau_evidence * tau_fresh",
    "decision_parameters": {
        "tau_lo": tau_lo, "tau_hi": tau_hi, "rho": rho, "c_obs": c_obs,
        "answer_target_error": target_risk, "calibration_confidence": confidence,
        "tau_lo_source": lo_source, "tau_hi_source": hi_source,
        "max_reobserve_steps": max(1, int(round(c_obs))),
    },
    "calibration_joint_rows": len(joint),
    "calibration_middle_rows": int(middle.sum()),
    "training_split": "predicted train memories only",
    "calibration_split": "scene-disjoint predicted calibration memories only",
    "audit_split_used": False,
}
(out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))

