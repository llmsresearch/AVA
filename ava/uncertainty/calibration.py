from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ava.utils.metrics import compute_expected_calibration_error


class IsotonicCalibrator:
    """
    Post-hoc calibration using isotonic regression.

    Maps raw confidence scores to calibrated probabilities.
    """

    def __init__(self) -> None:
        self.regressor: Optional[IsotonicRegression] = None
        self.fitted = False

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

