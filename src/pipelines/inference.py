import time
import numpy as np
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel


@dataclass
class MLFlowDeploymentLoaderStepParameters:
    """MLflow model loader parameters.

    Attributes:
        model_uri (str): MLflow model URI, for example:
            - runs:/<run_id>/model
            - models:/<model_name>/Production
            - local path to an exported MLflow model directory
    """
    model_uri: str


def prediction_service_loader(
    params: MLFlowDeploymentLoaderStepParameters,
) -> PyFuncModel:
    """Load an MLflow model for inference."""

    return mlflow.pyfunc.load_model(params.model_uri)


def predictor(
    model: PyFuncModel,
    data: np.ndarray,
) -> np.ndarray:
    """Run inference against an MLflow model.

    Args:
        model (PyFuncModel): Loaded MLflow model.
        data (np.ndarray): Image formatted as an array

    Returns:
        (np.ndarray) Prediction
    """
    time.sleep(1)
    prediction = model.predict(data)
    prediction = np.asarray(prediction)
    if prediction.ndim > 1 and prediction.shape[1] > 0:
        return prediction[:, 0]
    return prediction.reshape(-1)
