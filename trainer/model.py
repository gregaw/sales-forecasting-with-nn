"""Sales Forecast as a Neural Network model."""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

LOOK_BACK = 20

FEATURES_CONT = ['Open','Promo','DayOfWeek', 'StoreType']
FEATURES_WINDOW = ['Sales']
# NOTE: dependent variable will be always in last column. It seems to be required if we use OneHotEncoder
FEATURES_ALL = FEATURES_CONT + FEATURES_WINDOW

FEATURE_SIZE = LOOK_BACK + len(FEATURES_CONT)


def get_input_shape(dataset):
    """ Number of columns is known only after apply OneHotEncoder."""
    return (LOOK_BACK + dataset[0].size - 1,)


def model_fn(input_shape):
    """Create a Keras Sequential model with layers."""
    model = models.Sequential()
    model.add(layers.Dense(12, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(12, activation='relu'))
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
        row = dataset[i:(i + look_back), -1] # y is always in last column
        row = np.append(row, dataset[i + look_back, 0:-1])  # appending CONT
        x.append(row)
        y.append(dataset[i + look_back, -1])
    return np.array(x), np.array(y)


def get_all_data(input_files):
    """ Returns rows from all input_files """
    values = [_read_raw(input_file) for input_file in input_files]
    full_dataset = np.concatenate(values)
    return full_dataset


def build_onehotencoder(column_names, full_dataset):
    """ Returns fitted OneHotEncoder for each column on column_names list """
    col_indexes = []
    for col_name in column_names:
        if col_name in FEATURES_ALL: 
            col_indexes.append(FEATURES_ALL.index(col_name))
    
    if len(col_indexes) > 0:
        onehotencoder = OneHotEncoder(categorical_features=col_indexes)
        onehotencoder.fit(full_dataset)
        return onehotencoder
    else:
        return None


def getOneHotEncodedData(onehotencoder, dataset):
    """ 
    Encodes and return data using previously fitted onehotencoder
    TODO: it avoids Dummy Variable Trap only if used for one column only and 
        assumes that Dummy Variables are always appended at the beginning.
    """ 
    if onehotencoder is not None:
        data = onehotencoder.transform(dataset).toarray()
        # avoid dummy variable trap (assuming we have only on categorical column...)
        return data[:,1:]
    else:
        return dataset
    
    
def build_labelencoder(column_name, full_dataset):
    """ Returns LabelEncoder for given column_name """
    if column_name in FEATURES_ALL:
        labelencoder = LabelEncoder()
        index = FEATURES_ALL.index(column_name)
        labelencoder = labelencoder.fit(full_dataset[:,index])
        return labelencoder
    else:
        return None    
    
def apply_labelencoder(column_name, labelencoder, dataset):
    """ Applies previously fitted labelencoder for column_name in dataset """
    if labelencoder is not None:
        index = FEATURES_ALL.index(column_name)
        dataset[:,index] = labelencoder.transform(dataset[:,index])
    return
   

def build_scaler(full_dataset):
    """builds scaler based on OneHotEncoded/Labeled data from all files """
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


def load_features(input_files, scaler, labelencoder_DayOfWeek, labelencoder_StoreType, labelencoder_Assortment, onehotencoder):
    """generate features
    :returns (x, y)
    """
    xAll = []
    yAll = []

    for input_file in input_files:
        data = get_all_data([input_file])
        apply_labelencoder('DayOfWeek', labelencoder_DayOfWeek, data)
        apply_labelencoder('StoreType', labelencoder_StoreType, data)
        apply_labelencoder('Assortment', labelencoder_Assortment, data)
        data = getOneHotEncodedData(onehotencoder, data)
        data = scaler.transform(data)
        x, y = create_windows(data, LOOK_BACK)
        xAll = x if len(xAll) == 0 else np.concatenate((xAll, x))
        yAll = y if len(yAll) == 0 else np.concatenate((yAll, y))
    return xAll, yAll


def _read_raw(input_file):
    df = pd.read_csv(tf.gfile.Open(input_file), parse_dates=['Date'], dtype={'StateHoliday': np.str, 'StoreType': np.str})

    # consider only data when shop is opened?
    #values = df.loc[df['Open'] == 1][FEATURES_ALL].values
    values = df[FEATURES_ALL].values
    
    # convertion cannot be applied until categorical columns are not labelencoded
    # values = values.astype('float32')

    return values
