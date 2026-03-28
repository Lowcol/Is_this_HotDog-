import logging
import datetime
import numpy as np
from pathlib import Path

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Rescaling
)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.DEBUG)


class HotdogClassifier:
    def __init__(self, args):
        self.args = args

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray):
        logging.info("Constructing model...")

        tf.keras.utils.set_random_seed(self.args.get('random_seed', 0))

        dropout_rate = self.args.get('dropout', 0.2)

        output_units = int(self.args.get('dense_3_units', 1))
        if output_units == 1:
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            output_activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.args['image_size'], self.args['image_size'], 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        cnn2d = Sequential([
            tf.keras.layers.InputLayer(shape=(self.args['image_size'], self.args['image_size'], 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            Rescaling(scale=1.0 / 127.5, offset=-1.0),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(dropout_rate),
            Dense(units=self.args.get('dense_1_units', 256), activation='relu'),
            Dropout(dropout_rate),
            Dense(units=self.args.get('dense_2_units', 64), activation='relu'),
            Dropout(dropout_rate),
            Dense(units=output_units, activation=output_activation)
        ])

        cnn2d.compile(loss=loss,
                      optimizer=Adam(learning_rate=self.args['learning_rate']),
                      metrics=['accuracy'])
        logging.info("Model compiled.")

        logging.info("Fitting model...")

        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        cnn2d.fit(
            x=X_train,
            y=y_train,
            epochs=self.args['num_epochs'],
            batch_size=self.args['batch_size'],
            validation_split=self.args['val_split'],
        )
        logging.info("Model successfully fit.")
        return cnn2d

    def save(self, model):
        """Save model.

        Args:
            model: Trained tf.keras.model
        """
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path(self.args['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(output_dir / f"model-{time}.keras")

        tf.keras.models.save_model(model, filepath=out_path)
        logging.info(f"Model saved to {out_path}.")
