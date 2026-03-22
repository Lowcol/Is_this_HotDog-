import yaml
import logging
from typing import Optional

import mlflow
import numpy as np
import tensorflow as tf

from model.evaluator import Evaluation

logging.basicConfig(level=logging.DEBUG)


def evaluate_model(trained_model: tf.keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   experiment_name: str = "hotdog-classifier") -> tuple[float, float, float]:
    """Evaluates the model performance on the test set.
    Logs the model performance to MLFlow.

    Args:
        trained_model (tf.keras.Model): Trained tf.keras model
        X_test (np.ndarray): Array of test images
        y_test (np.ndarray): Array of test labels
        experiment_name (str): MLflow experiment name

    Returns:
        (float, float, float): precision, recall, f1 scores

    Raises:
        Exception if any of the metrics calculations fail
    """
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    classification_cutoff = configs.get('classification_cutoff', 0.5)

    mlflow.set_experiment(experiment_name)
    started_run: Optional[mlflow.ActiveRun] = None
    if mlflow.active_run() is None:
        started_run = mlflow.start_run(run_name="evaluate-hotdog-classifier")

    logging.info("Beginning model evaluation...")

    try:
        logging.info("Predicting on the test set...")
        # Support both historical one-hot labels (N, 2) and binary labels (N,).
        if y_test.ndim > 1:
            y_test = y_test[:, 0]
        y_test = y_test.astype(int)
        evaluation = Evaluation()

        raw_prediction = trained_model.predict(X_test)
        
        if raw_prediction.shape[1] == 1:
            prediction = raw_prediction[:, 0]
        else:
            prediction = raw_prediction[:, 1]
            
        prediction = np.where(prediction > classification_cutoff, 1, 0)

        logging.info(f"Calculating metrics with cut-off {classification_cutoff}...")
        precision = evaluation.precision(y_test, prediction)
        mlflow.log_metric("test_precision", precision)

        recall = evaluation.recall(y_test, prediction)
        mlflow.log_metric("test_recall", recall)

        f1 = evaluation.f1(y_test, prediction)
        mlflow.log_metric("test_f1", f1)

        logging.info("Model evaluation done.")

        if started_run is not None:
            mlflow.end_run()

        return precision, recall, f1

    except Exception as e:
        logging.error(e)
        if started_run is not None:
            mlflow.end_run(status="FAILED")
        raise e