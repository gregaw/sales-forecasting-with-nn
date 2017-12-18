import pandas as pd
import numpy as np

input_file = "data/train.csv"
stores_input_file = "data/store.csv"

dftrain = pd.read_csv(input_file, parse_dates=['Date'], dtype={'StateHoliday': np.str})
dfstore = pd.read_csv(stores_input_file)

# possible shortcut for labelencode
map_char = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3
        }

# create train/eval files for each store
for index, row in dfstore.iterrows():
    store_id = str(row['Store'])
    finalRowsDF = dftrain.loc[dftrain['Store'] == row['Store']]
    # we want to start with earlier first
    finalRowsDF = finalRowsDF.sort_values(by=['Date'], ascending=True)

    #add info about the store
    #sellStoreDF['Assortment'] = map_char[row['Assortment']]
    #sellStoreDF['StoreType'] = map_char[row['StoreType']]
    finalRowsDF['Assortment'] = row['Assortment']
    finalRowsDF['StoreType'] = row['StoreType']

    train_size = int(0.7 * finalRowsDF.shape[0])
    finalRowsDF[0:train_size].to_csv('data/train' + store_id + '.csv', index=False)
    finalRowsDF[train_size:].to_csv('data/eval' + store_id + '.csv', index=False)
