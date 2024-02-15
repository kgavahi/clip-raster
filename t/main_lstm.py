# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:26:02 2024

@author: kgavahi
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
import numpy as np

# Read the CSV file
input_file_path = 'merged_hysets_daymet_GLDAS_AMSR_snowdas_UAZ_MERRA2_hysets_09112200.csv'
df = pd.read_csv(input_file_path)


# Select the desired columns
# always put the target at the last col
desired_columns = ['date', 'swe_daymet', 'swe_UAZ', 'swe_GLDAS', 'streamflow']
df = df[desired_columns]

df['swe_daymet'] = np.arange(len(df))
df['swe_UAZ'] = np.arange(len(df))*10
df['swe_GLDAS'] = np.arange(len(df))*100
df['streamflow'] = np.arange(len(df))*1000




# import seaborn as sns
# sns.violinplot(data=df)


# Function to create sequences for LSTM
def create_sequences(data, n_past, n_future):
    X, y = [], []
    # pointer to select columns
    p = data.shape[1]
    for i in range(len(data) - n_past - n_future + 1):
        
        # Target is also an input
        #X.append(data.iloc[i:i + n_past, :p])
        #y.append(data.iloc[i + n_past:i + n_past + n_future, p-1])
        
        # Target is not included in the input
        X.append(data.iloc[i:i + n_past, :p-1])
        y.append(data.iloc[i + n_past:i + n_past + n_future, p-1])        
            
        
    return np.array(X), np.array(y)

# Encoder-Decoder LSTM
def build_model_ED(train_x, train_y, val_x, val_y):
    
 	# define parameters
 	verbose, epochs, batch_size = 0, 20, 16
 	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

 	# define model
 	model = Sequential()
 	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
 	model.add(RepeatVector(n_outputs))
 	model.add(LSTM(200, activation='relu', return_sequences=True))
 	model.add(TimeDistributed(Dense(100, activation='relu')))
 	model.add(TimeDistributed(Dense(1)))
 	model.compile(loss='mse', optimizer='adam')
 	# fit network
 	history = model.fit(train_x, train_y, 
                      validation_data=(val_x, val_y),
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose,
                      )
 	print('im Encoder-Decoder LSTM')
 	return model, history
# Naive LSTM
def build_model_naive(train_x, train_y, val_x, val_y):

    # define parameters
    verbose, epochs, batch_size = 0, 100, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y, 
                      validation_data=(val_x, val_y),
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose,
                      )
    print('im Naive')
    return model, history
def drop_nans(X_seq_tr, y_seq_tr):
    mask_x = np.isnan(X_seq_tr).any(axis=(1,2))
    mask_y = np.isnan(y_seq_tr).any(axis=(1,))
    mask = np.logical_or(mask_x, mask_y)
    
    X_seq_tr = X_seq_tr[~mask]
    y_seq_tr = y_seq_tr[~mask]    
    
    return X_seq_tr, y_seq_tr

# # load the new file
# df = pd.read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

# # Select the desired columns
# # always put the target at the last col
# desired_columns = ['Global_active_power']
# df = df[desired_columns]

# Read the CSV file
input_file_path = 'merged_hysets_daymet_GLDAS_AMSR_snowdas_UAZ_MERRA2_hysets_09112200.csv'
df = pd.read_csv(input_file_path)

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Select the desired columns
# always put the target at the last col
#desired_columns = ['swe_daymet', 'swe_UAZ', 'swe_GLDAS', 'streamflow']
#df = df[desired_columns]

desired_columns = ['snow_depth_water_equivalent_mean', 'surface_net_solar_radiation_mean',
       'surface_net_thermal_radiation_mean', 'surface_pressure_mean',
       'temperature_2m_mean', 'dewpoint_temperature_2m_mean',
       'u_component_of_wind_10m_mean', 'v_component_of_wind_10m_mean',
       'volumetric_soil_water_layer_1_mean',
       'volumetric_soil_water_layer_2_mean',
       'volumetric_soil_water_layer_3_mean',
       'volumetric_soil_water_layer_4_mean', 'snow_depth_water_equivalent_min',
       'surface_net_thermal_radiation_min',
       'surface_pressure_min', 'temperature_2m_min',
       'dewpoint_temperature_2m_min', 'u_component_of_wind_10m_min',
       'v_component_of_wind_10m_min', 'volumetric_soil_water_layer_1_min',
       'volumetric_soil_water_layer_2_min',
       'volumetric_soil_water_layer_3_min',
       'volumetric_soil_water_layer_4_min', 'snow_depth_water_equivalent_max',
       'surface_net_solar_radiation_max', 'surface_net_thermal_radiation_max',
       'surface_pressure_max', 'temperature_2m_max',
       'dewpoint_temperature_2m_max', 'u_component_of_wind_10m_max',
       'v_component_of_wind_10m_max', 'volumetric_soil_water_layer_1_max',
       'volumetric_soil_water_layer_2_max',
       'volumetric_soil_water_layer_3_max',
       'volumetric_soil_water_layer_4_max', 'total_precipitation_sum',
       'potential_evaporation_sum', 'swe_daymet', 'swe_GLDAS',
       'swe_UAZ', 'streamflow']
df = df[desired_columns]


# plot some columns
import matplotlib.pyplot as plt
plot_cols = ['streamflow']
plot_features = df[plot_cols]
plot_features.index = df.index
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = df.index[:480]
_ = plot_features.plot(subplots=True)


#print(df.describe().transpose())

# train test val split (val split will be done inside the model)
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# peek at the distribution of the features
import seaborn as sns
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)


n_past, n_future = 14, 1
X_seq_tr, y_seq_tr = create_sequences(train_df, n_past, n_future)
X_seq_val, y_seq_val = create_sequences(val_df, n_past, n_future)
X_seq_test, y_seq_test = create_sequences(test_df, n_past, n_future)

# drop nans here not at the begining to maintain the sequence
X_seq_tr, y_seq_tr = drop_nans(X_seq_tr, y_seq_tr)
X_seq_val, y_seq_val = drop_nans(X_seq_val, y_seq_val)
X_seq_test, y_seq_test = drop_nans(X_seq_test, y_seq_test)



model, history = build_model_naive(X_seq_tr, y_seq_tr, X_seq_val, y_seq_val)

# Check validation loss for overfitting
#print(history.history.keys())
#print(history.history['val_loss'])
plt.pause(0.1)
__ = plt.plot(history.history['loss'])
__ = plt.plot(history.history['val_loss'])

yhat = model.predict(X_seq_test, verbose=0)


yhat = (yhat * train_std['streamflow']) + train_mean['streamflow']
yhat = yhat.reshape(yhat.shape[0], 1)

y_seq_test = (y_seq_test * train_std['streamflow']) + train_mean['streamflow']

from sklearn.metrics import mean_squared_error, r2_score
# Calculate R2 and MSE
r2  = r2_score(yhat, y_seq_test)
mse = mean_squared_error(yhat, y_seq_test)

print('r2:', r2, 'mse', mse)

plt.pause(0.1)
__ = plt.plot(yhat)
__ = plt.plot(y_seq_test)











