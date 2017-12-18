# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob

import keras
import os
from keras.models import load_model
from tensorflow.python.lib.io import file_io
import tensorflow as tf
import numpy as np

from trainer import model

CHECKPOINT_PATH = 'checkpoint.{epoch:06d}.hdf5'
MODEL_FILENAME = 'sales_forecaster.hdf5'


class TensorBoardMetricsLogger:
    """Log your own metrics to be shown by TensorBoard"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)

    def append(self, metrics_dict, epoch):
        for (name, value) in metrics_dict.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)

        self.writer.flush()

    def close(self):
        self.writer.close()


class ContinuousEval(keras.callbacks.Callback):
    """
    Continuous eval callback to evaluate the checkpoint once every so many epochs.
    Saves
    Saves eval predictions to 'preds' folder and Tensorboard eval metrics to 'val_logs' folder.
    """

    def __init__(self,
                 eval_frequency,
                 eval_files,
                 learning_rate,
                 job_dir,
                 scaler,
                 labelencoder_DayOfWeek,
                 labelencoder_StoreType,
                 labelencoder_Assortment,
                 onehotencoder,
                 steps=1000):
        self.scaler = scaler
        self.labelencoder_DayOfWeek = labelencoder_DayOfWeek
        self.labelencoder_StoreType = labelencoder_StoreType
        self.labelencoder_Assortment = labelencoder_Assortment
        self.onehotencoder = onehotencoder
        self.eval_files = eval_files
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.steps = steps
        self.tf_logger = TensorBoardMetricsLogger(os.path.join(job_dir, 'val_logs'))
        self.epochs_since_last_save = 0
        os.makedirs(os.path.join(self.job_dir, 'preds'))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.tf_logger.close()

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.eval_frequency:
            self.epochs_since_last_save = 0
            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith("gs://"):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                forecast_model = load_model(checkpoints[-1])
                forecast_model = model.compile_model(forecast_model)
                x, y = model.load_features(self.eval_files, self.scaler, self.labelencoder_DayOfWeek, self.labelencoder_StoreType, self.labelencoder_Assortment, self.onehotencoder)
                metrics = forecast_model.evaluate(x, y)
                print('\n*** Evaluation epoch[{}] metrics {} {}'.format(
                    epoch, metrics, forecast_model.metrics_names))

                y_hat = forecast_model.predict(x)
                y_hat = model.invert_scale_sales(y_hat, self.scaler)
                np.savetxt(os.path.join(self.job_dir, 'preds/yhat_{:06d}.txt'.format(epoch)), y_hat)

                self.tf_logger.append(
                    metrics_dict={name: value for (name, value) in zip(forecast_model.metrics_names, metrics)},
                    epoch=epoch
                )

                if self.job_dir.startswith("gs://"):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print('\n*** Evaluation epoch[{}] (no checkpoints found)'.format(epoch))


def dispatch(train_files,
             eval_files,
             job_dir,
             learning_rate,
             eval_frequency,
             num_epochs,
             checkpoint_epochs):

    # setting the seed for reproducibility
    np.random.seed(13)

    # get all data and build labelencoder and onehotencoder
    full_dataset = model.get_all_data(train_files + eval_files)
    
    # convert values in categorical columns to numerical 0-n
    labelencoder_DayOfWeek = model.build_labelencoder('DayOfWeek', full_dataset)
    labelencoder_StoreType = model.build_labelencoder('StoreType', full_dataset)
    labelencoder_Assortment = model.build_labelencoder('Assortment', full_dataset)
    
    # NOTE: apply label encoders before build onehotencoder
    model.apply_labelencoder('DayOfWeek', labelencoder_DayOfWeek, full_dataset)
    model.apply_labelencoder('StoreType', labelencoder_StoreType, full_dataset)
    model.apply_labelencoder('Assortment', labelencoder_Assortment, full_dataset)

    # DayOfWeek should be considered as categorical data and not as numerical
    onehotencoder = model.build_onehotencoder(['DayOfWeek', 'StoreType', 'Assortment'], full_dataset)
    #onehotencoder_DayOfWeek = model.build_onehotencoder_DayOfWeek(full_dataset)
    full_dataset = model.getOneHotEncodedData(onehotencoder, full_dataset)

    # NOTE: must be called after apply Label- and OneHot- Encoder
    scaler = model.build_scaler(full_dataset)

    # finally we can create our model
    input_data_shape = model.get_input_shape(full_dataset)
    forecast_model = model.model_fn(input_data_shape)

    try:
        os.makedirs(job_dir)
    except Exception as e:
        print(e)

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = CHECKPOINT_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1,
        period=checkpoint_epochs
    )

    # Continuous eval callback
    with ContinuousEval(eval_frequency,
                        eval_files,
                        learning_rate,
                        job_dir,
                        scaler,
                        labelencoder_DayOfWeek,
                        labelencoder_StoreType,
                        labelencoder_Assortment,
                        onehotencoder) as evaluation:

        # Tensorboard logs callback
        tblog = keras.callbacks.TensorBoard(
            log_dir=os.path.join(job_dir, 'logs'),
            histogram_freq=0,
            write_graph=True,
            embeddings_freq=0)

        callbacks = [checkpoint, evaluation, tblog]

        x, y = model.load_features(train_files, scaler, labelencoder_DayOfWeek, labelencoder_StoreType, labelencoder_Assortment, onehotencoder)
        forecast_model.fit(
            x, y,
            epochs=num_epochs,
            callbacks=callbacks)

        # Unhappy hack to work around h5py not being able to write to GCS.
        # Force snapshots and saves to local filesystem, then copy them over to GCS.
        if job_dir.startswith("gs://"):
            forecast_model.save(MODEL_FILENAME)
            copy_file_to_gcs(job_dir, MODEL_FILENAME)
        else:
            forecast_model.save(os.path.join(job_dir, MODEL_FILENAME))


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--eval-frequency',
                        type=int,
                        default=10,
                        help='Perform one evaluation per n epochs')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--checkpoint-epochs',
                        type=int,
                        default=5,
                        help='Checkpoint per n training epochs')
    parsed_args, unknown = parser.parse_known_args(args)

    return parsed_args


if __name__ == "__main__":
    parsed_args = parse_args()
    dispatch(**parsed_args.__dict__)
