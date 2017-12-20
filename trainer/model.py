"""Sales Forecast as a Neural Network model."""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler

LOOK_BACK = 20

FEATURES_CONT = []
FEATURES_WINDOW = ['Sales']
FEATURES_ALL = FEATURES_WINDOW + FEATURES_CONT

FEATURE_SIZE = LOOK_BACK + len(FEATURES_CONT)


def model_fn():
    """Create a Keras Sequential model with layers."""
    model = models.Sequential()
    model.add(layers.Dense(30, input_shape=(FEATURE_SIZE,), activation='relu'))
    model.add(layers.Dense(5, input_shape=(FEATURE_SIZE,), activation='relu'))
    model.add(layers.Dense(1))

    compile_model(model)

    return model


def compile_model(model):
    """Compiles the model - either created or loaded"""
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


def create_windows(dataset, look_back=1):
    """
    create windows of data
    :param dataset: first column (sales) is to be have row look back, rest is just appended
    :param look_back:
    :return: ( x:(n-LB-1, LB+1), y:(n-LB-1, 1) )
    """
    x, y = [], []
    for i in range(len(dataset) - look_back):
        row = dataset[i:(i + look_back), 0]
        row = np.append(row, dataset[i + look_back, 1:])  # appending CONT
        x.append(row)
        y.append(dataset[i + look_back, 0])
    return np.array(x), np.array(y)


def build_scaler(input_files):
    """builds scaler based on all the input_files"""
    values = [_read_raw(input_file) for input_file in input_files]
    full_dataset = np.concatenate(values)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(full_dataset)

    return scaler


def invert_scale_sales(sales_vector, scaler):
    """(n,1) -> (n,features) -> invert_scale -> (n,1)"""
    # demo: hardcoding sales index
    sales_index = 0
    inverted_sales = sales_vector - scaler.min_[sales_index]
    inverted_sales /= scaler.scale_[sales_index]

    return inverted_sales


def load_features(input_files, scaler):
    """generate features
    :returns (x, y)
    """
    # demo: we just use one file
    input_file = input_files[0]

    data = _read_raw(input_file)

    data = scaler.transform(data)

    x, y = create_windows(data, LOOK_BACK)

    return x, y


def _read_raw(input_file):
    df = pd.read_csv(tf.gfile.Open(input_file), parse_dates=['Date'], dtype={'StateHoliday': np.str})

    values = df[FEATURES_ALL].values
    values = values.astype('float32')

    return values
