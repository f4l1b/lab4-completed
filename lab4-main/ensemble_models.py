
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:

    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=data.random_seed,
    )
    model.fit(data.train_features, np.asarray(data.train_target).ravel())

    probabilities = model.predict_proba(data.test_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    y_test = np.asarray(data.test_target).ravel()

    return EnsembleResult(
        model_name="Gradient Boosting (250 estimators, learning_rate=0.08, max_depth=1)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
