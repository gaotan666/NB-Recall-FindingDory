"""Load the trained writer and calibrated evidence/freshness trust heads."""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from .memory import MemoryGraph, MemoryNode

WRITER_FEATURES = ("has_prior", "same_entity", "same_acquisition_group", "same_sha256",
                   "pose_delta_m", "age_steps", "log_pixels", "depth_m")
TRUST_FEATURES = ("semantic_match", "provenance_freshness", "version_consistency",
                  "conflict_degree", "support_coverage", "source_reliability")


def sigmoid(x):
    return float(1 / (1 + np.exp(-np.clip(x, -50, 50))))


class WriterController:
    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / "writer.joblib")
        self.report = json.loads((model_dir / "report.json").read_text())
        if tuple(self.report["feature_names"]) != WRITER_FEATURES:
            raise ValueError("Writer feature schema mismatch")
        self.min_confidence = self.report["min_write_confidence"]
        if self.min_confidence is None:
            raise ValueError("Writer lacks a calibrated confidence threshold")

    def decide(self, candidate: MemoryNode, prior: Optional[MemoryNode]):
        cp = candidate.provenance[0]
        pp = prior.provenance[-1] if prior else None
        same_entity = bool(prior and candidate.entity_id == prior.entity_id)
        same_group = bool(prior and candidate.acquisition_group == prior.acquisition_group)
        same_sha = bool(pp and cp.get("sha256") == pp.get("sha256"))
        pose_delta = math.dist(candidate.content, prior.content) if prior else 0.0
        age = cp["time"] - pp["time"] if pp else 0
        x = np.array([[int(prior is not None), int(same_entity), int(same_group),
                       int(same_sha), pose_delta, age,
                       math.log1p(candidate.support["pixel_count"]),
                       float(candidate.support["depth_m"])]])
        probabilities = self.model.predict_proba(x)[0]
        index = int(np.argmax(probabilities))
        operation = str(self.model.classes_[index])
        confidence = float(probabilities[index])
        if confidence < self.min_confidence:
            return "DROP", confidence, "uncertain_writer"
        return operation, confidence, "model"

    def insert(self, graph: MemoryGraph, candidate: MemoryNode):
        versions = [n for n in graph.nodes.values() if n.type == candidate.type and
                    n.entity_id == candidate.entity_id and n.attribute == candidate.attribute and
                    n.support["scope"]["scene"] == candidate.support["scope"]["scene"]]
        prior = max(versions, key=lambda n: n.validity["t_valid_from"]) if versions else None
        operation, confidence, source = self.decide(candidate, prior)
        if operation in {"MERGE", "UPDATE"} and prior is None:
            operation, source = "DROP", "invalid_no_prior"
        if operation == "WRITE" and prior is not None:
            # Multiple versions of one Fact must be linked or explicitly disputed.
            operation, source = "DROP", "unlinked_fact_version"
        if operation != "DROP":
            graph.apply(operation, candidate, prior.id if prior else None)
        return {"operation": operation, "confidence": confidence, "source": source}


class TrustScorer:
    def __init__(self, model_dir: Path):
        self.evidence = joblib.load(model_dir / "tau_evidence.joblib")
        self.fresh = joblib.load(model_dir / "tau_fresh.joblib")
        report = json.loads((model_dir / "report.json").read_text())
        if tuple(report["feature_names"]) != TRUST_FEATURES:
            raise ValueError("Trust feature schema mismatch")
        self.temp_e = report["temperature"]["tau_evidence"]
        self.temp_f = report["temperature"]["tau_fresh"]
        self.tau_lo = report["decision_parameters"]["tau_lo"]
        self.tau_hi = report["decision_parameters"]["tau_hi"]

    def score(self, features: Dict[str, float]):
        x = np.array([[float(features[name]) for name in TRUST_FEATURES]])
        tau_evidence = sigmoid(self.evidence.decision_function(x)[0] / self.temp_e)
        tau_fresh = sigmoid(self.fresh.decision_function(x)[0] / self.temp_f)
        return {"tau_evidence": tau_evidence, "tau_fresh": tau_fresh,
                "tau": tau_evidence * tau_fresh}


@dataclass(frozen=True)
class GateParameters:
    tau_lo: float
    tau_hi: float
    rho: float
    c_obs: float
    max_reobserve_steps: int

    def __post_init__(self):
        if not 0 <= self.tau_lo < self.tau_hi <= 1:
            raise ValueError("Require 0 <= tau_lo < tau_hi <= 1")
        if not 0 < self.rho <= 1 or self.c_obs < 0 or self.max_reobserve_steps < 1:
            raise ValueError("Re-observation information/cost/budget must be measured and valid")


def decide_action(tau: float, parameters: GateParameters):
    if tau >= parameters.tau_hi:
        return "answer"
    if tau >= parameters.tau_lo:
        return "re-observe"
    return "abstain"
