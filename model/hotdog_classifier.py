import logging
import datetime
import numpy as np
from pathlib import Path

# TODO random seed
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization
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

        cnn2d = Sequential()
        cnn2d.add(
            Conv2D(
                filters=self.args.get('conv_1_filters', 32),
                kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
                activation='relu',
                input_shape=(self.args['image_size'], self.args['image_size'], 3)))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Conv2D(
            filters=self.args.get('conv_2_filters', 64),
            kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
            activation='relu'))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Conv2D(
            filters=self.args.get('conv_3_filters', 128),
            kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
            activation='relu'))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Dropout(
            self.args['dropout']))
        cnn2d.add(Flatten())
        cnn2d.add(Dense(
            units=self.args.get('dense_1_units', 128),
            activation='relu'))
        cnn2d.add(Dropout(
            self.args['dropout']))
        cnn2d.add(Dense(
            units=1,
            activation='sigmoid'))

        cnn2d.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=self.args['learning_rate']),
                      metrics=['accuracy'])
        logging.info("Model compiled.")

        logging.info("Fitting model...")
        cnn2d.fit(
            x=X_train,
            y=y_train,
            epochs=self.args['num_epochs'],
            batch_size=self.args['batch_size'],
            validation_split=self.args['val_split']
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
        out_path = str(output_dir / f"model-{time}.h5")

        tf.keras.models.save_model(
            model,
            filepath=out_path,
            save_format='h5'
        )
        logging.info(f"Model saved to {out_path}.")