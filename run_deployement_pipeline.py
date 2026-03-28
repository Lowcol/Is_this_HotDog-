import yaml
import logging
import os
import tensorflow as tf

import mlflow
from mlflow.models import infer_signature
from src.pipelines.import_data import import_data
from src.pipelines.train_model import train_model
from src.pipelines.evaluate_model import evaluate_model
from src.pipelines.deploy_model import DeploymentTriggerConfig, deployment_trigger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_main():
    with open('config/pipeline_config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    logger.info("Starting continuous deployment pipeline...")

    # Step 1: Import data
    logger.info("Step 1: Importing data...")
    X_train, y_train, X_test, y_test = import_data()
    logger.info(f"Data imported. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Step 2: Check for existing model or train a new one
    logger.info("Step 2: Checking for existing trained model...")
    models_dir = configs.get('output_path', 'models/')
    
    # Check if directory exists and find the latest .keras model
    latest_model_path = None
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        if model_files:
            # Sort files by name (which includes timestamp format like model-2026-03-22-15-13-01.keras)
            latest_model_file = sorted(model_files)[-1]
            latest_model_path = os.path.join(models_dir, latest_model_file)
    
    if latest_model_path:
        logger.info(f"Existing model found: {latest_model_path}")
        logger.info("Loading existing model, skipping training phase...")
        trained_model = tf.keras.models.load_model(latest_model_path)
    else:
        logger.info("No existing model found. Proceeding to train a new model...")
        trained_model = train_model(X_train, y_train)
        logger.info("Model training completed.")

    # Step 3: Evaluate model
    logger.info("Step 3: Evaluating model...")
    precision, recall, f1 = evaluate_model(trained_model, X_test, y_test)
    logger.info(f"Model evaluation completed. Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Step 4: Check deployment trigger
    logger.info("Step 4: Checking deployment criteria...")
    trigger_config = DeploymentTriggerConfig(
        min_precision=configs['min_precision'],
        min_recall=configs['min_recall']
    )
    should_deploy = deployment_trigger(precision, recall, trigger_config)

    if should_deploy:
        logger.info("✓ Deployment criteria met. Deploying model...")
        # Step 5: Deploy to MLflow
        input_example = X_test[:1].astype('float32')
        output_example = trained_model.predict(input_example, verbose=0)
        signature = infer_signature(input_example, output_example)
        with mlflow.start_run(run_name="model-deployment"):
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.tensorflow.log_model(
                model=trained_model,
                name="hotdog_classifier_model",
                signature=signature,
                registered_model_name="hotdog-classifier"
            )
        logger.info("✓ Model successfully deployed to MLflow.")
        print("\nModel deployment successful!")
    else:
        logger.warning(f"✗ Deployment criteria not met.")
        logger.warning(f"  Required: precision > {trigger_config.min_precision}, recall > {trigger_config.min_recall}")
        logger.warning(f"  Got: precision = {precision:.4f}, recall = {recall:.4f}")
        print("\nModel did not meet deployment criteria. Skipping deployment.")


if __name__ == "__main__":
    run_main()
