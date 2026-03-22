import yaml
import logging
import argparse
import sys
import importlib.util
from typing import Optional
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import tensorflow as tf
from mlflow.models import infer_signature

from model.hotdog_classifier import HotdogClassifier

logging.basicConfig(level=logging.DEBUG)


def _load_train_arrays(train_dir: str,
                       image_size: int,
                       batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Loads training data from a folder structure and returns numpy arrays.

    Expected directory layout:
        train_dir/
            hotdog/
            not_hotdog/
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    # The models handle preprocessing directly now

    X_batches = []
    y_batches = []
    for X_batch, y_batch in dataset:
        X_batches.append(X_batch.numpy())
        y_batches.append(y_batch.numpy())

    X_train = np.concatenate(X_batches, axis=0)
    y_train = np.concatenate(y_batches, axis=0)
    return X_train, y_train


def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                experiment_name: str = "hotdog-classifier",
                enable_autolog: bool = True) -> tf.keras.Model:
    """Trains the hotdog classifier model, logs the run to MLFlow,
    and saves the trained model locally.

    Args:
        X_train (np.ndarray): Array of train images
        y_train (np.ndarray): Array of training labels
        experiment_name (str): MLflow experiment name

    Returns:
        (tf.keras.Model): Trained model
    """
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    mlflow.set_experiment(experiment_name)
    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    use_tf_autolog = enable_autolog and tensorboard_available

    if use_tf_autolog:
        mlflow.tensorflow.autolog(log_models=False)
    else:
        logging.warning(
            "MLflow TensorFlow autologging disabled (TensorBoard missing or disabled). "
            "Using manual MLflow logging."
        )

    hotdog_classifier = HotdogClassifier(configs)

    started_run: Optional[mlflow.ActiveRun] = None
    if mlflow.active_run() is None:
        started_run = mlflow.start_run(run_name="train-hotdog-classifier")

    logging.info("Starting training...")
    model = hotdog_classifier.train(X_train, y_train)

    logging.info("Saving model...")
    hotdog_classifier.save(model)

    mlflow.log_params(configs)
    input_example = X_train[:1].astype('float32')
    output_example = model.predict(input_example, verbose=0)
    signature = infer_signature(input_example, output_example)
    mlflow.tensorflow.log_model(model=model, name="model", signature=signature)

    logging.info("Done.")

    if started_run is not None:
        mlflow.end_run()

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the hotdog model.")
    parser.add_argument(
        "--train-dir",
        type=str,
        default="tests/data/train",
        help="Directory containing class folders (hotdog, not_hotdog).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="hotdog-classifier",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--disable-autolog",
        action="store_true",
        help="Disable MLflow TensorFlow autologging.",
    )
    args = parser.parse_args()

    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    logging.info("Loading training data from %s...", args.train_dir)
    X_train, y_train = _load_train_arrays(
        train_dir=args.train_dir,
        image_size=configs['image_size'],
        batch_size=configs['batch_size'],
    )

    train_model(
        X_train=X_train,
        y_train=y_train,
        experiment_name=args.experiment_name,
        enable_autolog=not args.disable_autolog,
    )


if __name__ == '__main__':
    main()