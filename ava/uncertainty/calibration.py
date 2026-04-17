from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ava.utils.metrics import compute_expected_calibration_error


class IsotonicCalibrator:
    """
    Post-hoc calibration using isotonic regression.

    Maps raw confidence scores to calibrated probabilities.
    Supports save/load for cross-dataset transfer experiments.
    """

    def __init__(self, source_dataset: str = "unknown") -> None:
        self.regressor: Optional[IsotonicRegression] = None
        self.fitted = False
        self.source_dataset = source_dataset
        self._training_stats: Dict[str, Any] = {}

    def fit(self, predicted_probs: List[float], actual_labels: List[bool]) -> None:
        """
        Fit calibrator to historical predictions.

        Args:
            predicted_probs: List of predicted probabilities [0,1]
            actual_labels: List of true labels (bool)
        """
        if len(predicted_probs) == 0:
            return

        self.regressor = IsotonicRegression(out_of_bounds="clip")
        y_true = np.array([float(x) for x in actual_labels])
        y_pred = np.array(predicted_probs)
        self.regressor.fit(y_pred, y_true)
        self.fitted = True

        # Store training statistics for analysis
        self._training_stats = {
            "n_samples": len(predicted_probs),
            "mean_pred": float(np.mean(y_pred)),
            "mean_true": float(np.mean(y_true)),
            "pred_range": [float(np.min(y_pred)), float(np.max(y_pred))],
        }

    def predict(self, confidence: float) -> float:
        """
        Transform raw confidence to calibrated probability.

        Args:
            confidence: Raw confidence score [0,1]

        Returns:
            Calibrated probability [0,1]
        """
        if not self.fitted or self.regressor is None:
            return confidence

        calibrated = self.regressor.predict([confidence])[0]
        return float(np.clip(calibrated, 0.0, 1.0))

    def save(self, path: str) -> None:
        """
        Save calibrator to file for later loading.

        Args:
            path: File path to save calibrator (will create .json)
        """
        if not self.fitted or self.regressor is None:
            raise ValueError("Cannot save unfitted calibrator")

        # Extract isotonic regression parameters
        # The regressor stores X_ and y_ arrays defining the piecewise function
        data = {
            "source_dataset": self.source_dataset,
            "fitted": self.fitted,
            "training_stats": self._training_stats,
            "X_thresholds": self.regressor.X_thresholds_.tolist(),
            "y_thresholds": self.regressor.y_thresholds_.tolist(),
            "X_min": float(self.regressor.X_min_),
            "X_max": float(self.regressor.X_max_),
            "increasing": bool(self.regressor.increasing_),
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        """
        Load calibrator from saved file.

        Args:
            path: File path to load calibrator from

        Returns:
            Loaded IsotonicCalibrator instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        calibrator = cls(source_dataset=data.get("source_dataset", "unknown"))
        calibrator.fitted = data["fitted"]
        calibrator._training_stats = data.get("training_stats", {})

        # Reconstruct isotonic regression
        calibrator.regressor = IsotonicRegression(out_of_bounds="clip")
        calibrator.regressor.X_thresholds_ = np.array(data["X_thresholds"])
        calibrator.regressor.y_thresholds_ = np.array(data["y_thresholds"])
        calibrator.regressor.X_min_ = data["X_min"]
        calibrator.regressor.X_max_ = data["X_max"]
        calibrator.regressor.increasing_ = data["increasing"]

        return calibrator

    def get_training_stats(self) -> Dict[str, Any]:
        """Return statistics from training data."""
        return self._training_stats.copy()


class PlattScalingCalibrator:
    """
    Post-hoc calibration using Platt scaling (logistic regression).
    """

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression()
        self.fitted = False

    def fit(self, predicted_probs: List[float], actual_labels: List[bool]) -> None:
        """Fit calibrator using logistic regression."""
        if len(predicted_probs) == 0:
            return

        y_true = np.array([int(x) for x in actual_labels])
        y_pred = np.array(predicted_probs).reshape(-1, 1)
        self.model.fit(y_pred, y_true)
        self.fitted = True

    def predict(self, confidence: float) -> float:
        """Transform raw confidence to calibrated probability."""
        if not self.fitted:
            return confidence

        calibrated = self.model.predict_proba([[confidence]])[0][1]
        return float(np.clip(calibrated, 0.0, 1.0))


def calibrate_predictions(
    predicted_probs: List[float],
    actual_labels: List[bool],
    method: str = "isotonic",
) -> Tuple[List[float], float, float]:
    """
    Calibrate predictions and return metrics.

    Args:
        predicted_probs: Raw predicted probabilities
        actual_labels: True labels
        method: "isotonic" or "platt"

    Returns:
        (calibrated_probs, ece_before, ece_after)
    """
    if method == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        calibrator = PlattScalingCalibrator()

    ece_before = compute_expected_calibration_error(predicted_probs, actual_labels)
    calibrator.fit(predicted_probs, actual_labels)
    calibrated_probs = [calibrator.predict(p) for p in predicted_probs]
    ece_after = compute_expected_calibration_error(calibrated_probs, actual_labels)

    return calibrated_probs, ece_before, ece_after

