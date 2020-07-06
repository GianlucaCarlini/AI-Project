# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:17:15 2020

@author: Gianluca
"""

### COVID-19

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Flatten, Dense, InputLayer, AveragePooling1D
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

''' DEFINITIONS '''

def MakeTimeSequence(array, seq_len = 4, forecast = 1):
    seq = []
    outcome = []
    for i in range(len(array) - seq_len - (forecast - 1)):
        for j in range(seq_len):
            seq.append(array[i + j])
        for k in range(forecast):
            outcome.append(array[i + seq_len + k])
            
    sequence = np.array(seq).reshape(-1, seq_len)
    future = np.array(outcome).reshape(-1, forecast)
    
    return sequence, future

def TotalCases(array):
    cases = []
    cases.append(array[0])
    for i in range(1, len(array)):
        cases.append(array[i] + cases[i-1])
    
    return np.array(cases)

def MeanSquaredError(Y_pred, Y_true):
    x = 0
    for i in range(len(Y_pred)):
        x += np.square(Y_pred[i] - Y_true[i])
    x = x/len(Y_pred)
    return x

''' 
---------------------------------------------------------------------------
'''

''' DATA WRANGLING'''

path = r'C:\Users\Gianluca\Desktop\Gianluca\Intelligenza_Artificiale\Progetto\Covid_World_Cases.csv'

data = pd.read_csv(path)

data.head(10)

Italy = data[data['countriesAndTerritories'] == 'Italy']

Italy = Italy.iloc[::-1]

daily_cases_italy = Italy[['dateRep', 'cases']].set_index('dateRep')
daily_deaths_italy = Italy[['dateRep', 'deaths']].set_index('dateRep')

total_italy = daily_cases_italy.join(daily_deaths_italy)

from_positive = total_italy[total_italy['cases'] > 5]

cases_array = from_positive['cases'].to_numpy()
days = np.arange(0, len(cases_array), 1).reshape(-1, 1)

total_cases = TotalCases(cases_array)

# plt.plot(days, total_cases)

# plt.title('Total Cases in Italy', fontsize = 18)
# plt.xlabel('Days', fontsize = 16)
# plt.ylabel('Cases', fontsize = 16)
# plt.tight_layout()
# plt.show()
# plt.savefig('Total_Cases_Italy.pgf')
# plt.close()

X, Y = MakeTimeSequence(total_cases, seq_len = 16, forecast = 2)

X = X.reshape(-1, 16, 1)
np.max(Y)
X = X/np.max(total_cases)
Y = Y/np.max(total_cases)

first_sixty_days = X[:66,:,:]
days_after = Y[:66,:]

validation_X = X[66:68,:,:] 
validation_Y = Y[66:68,:]

''' 
---------------------------------------------------------------------------
'''

''' MODEL '''

es = EarlyStopping(monitor = 'val_loss', patience = 200, mode = 'min')
baseline_mpc_save = ModelCheckpoint('Baseline_2D.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
baseline_csv = CSVLogger('Baseline_2D.csv')

Conv_mpc_save = ModelCheckpoint('Conv_Dense_2D.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
Conv_csv = CSVLogger('Conv_Dense_2D.csv')

FullyConv_mpc_save = ModelCheckpoint('FullyConv_2D.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
FullyConv_csv = CSVLogger('FullyConv_2D.csv')

opt = Adam(learning_rate = 0.0001)

baseline_model = Sequential()

baseline_model.add(Dense(50, activation = 'relu', input_shape = (16, 1)))
baseline_model.add(Flatten())
baseline_model.add(Dense(2))


baseline_model.compile(optimizer = opt, loss = 'mse')
baseline_model.summary()

baseline_model.fit(first_sixty_days, days_after, validation_data = (validation_X, validation_Y), 
                    epochs = 2000, callbacks = [es, baseline_mpc_save, baseline_csv], batch_size=16)


model = Sequential()
model.add(Conv1D(32, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 1, input_shape = (16, 1)))
model.add(Conv1D(32, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 2))
model.add(Conv1D(64, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 4))
model.add(Conv1D(64, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 8))

model.add(Flatten())
model.add(Dense(2))

model.compile(optimizer = opt, loss = 'mse')

model.summary()

model.fit(first_sixty_days, days_after, validation_data = (validation_X, validation_Y),
          epochs = 2000, callbacks = [es, Conv_mpc_save, Conv_csv], batch_size=16)



FullyConv = Sequential()
FullyConv.add(Conv1D(32, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 1, input_shape=(16,1)))
FullyConv.add(Conv1D(32, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 2))
FullyConv.add(Conv1D(64, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 4))
FullyConv.add(Conv1D(64, kernel_size = 2, activation = 'relu', padding = 'causal', dilation_rate = 8))
FullyConv.add(Conv1D(1, kernel_size = 2, activation = 'relu', padding = 'same'))
FullyConv.add(AveragePooling1D(pool_size = 8))
FullyConv.add(Flatten())

FullyConv.summary()

FullyConv.compile(optimizer = opt, loss = 'mse')

FullyConv.fit(first_sixty_days, days_after, validation_data = (validation_X, validation_Y),
              epochs = 2000, callbacks = [es, FullyConv_mpc_save, FullyConv_csv], batch_size=16)



''' 
---------------------------------------------------------------------------
'''

best_baseline = load_model(r'C:\Users\Gianluca\Desktop\Intelligenza_Artificiale\Progetto\Baseline_2D.mdl_wts.hdf5')
best_Conv_Dense = load_model(r'C:\Users\Gianluca\Desktop\Intelligenza_Artificiale\Progetto\Conv_Dense_2D.mdl_wts.hdf5')
best_FullyConv = load_model(r'C:\Users\Gianluca\Desktop\Intelligenza_Artificiale\Progetto\FullyConv_2D.mdl_wts.hdf5')


best_baseline_prediction = best_baseline.predict(validation_X)
best_Conv_Dense_prediction = best_Conv_Dense.predict(validation_X)
best_FullyConv_prediction = best_FullyConv.predict(validation_X)


best_baseline_mse = MeanSquaredError(best_baseline_prediction[1], validation_Y[1])
best_Conv_Dense_mse = MeanSquaredError(best_Conv_Dense_prediction[1], validation_Y[1])
best_FullyConv_mse = MeanSquaredError(best_FullyConv_prediction[1], validation_Y[1])

best_baseline_prediction *= np.max(total_cases)
best_Conv_Dense_prediction *= np.max(total_cases)
best_FullyConv_prediction *= np.max(total_cases)
validation_Y *= np.max(total_cases)

np.save('best_Conv_Dense_2D_prediction.npy', best_Conv_Dense_prediction)
np.save('best_baseline_2Dprediction.npy', best_baseline_prediction)
np.save('best_FullyConv_2Dprediction.npy', best_FullyConv_prediction)
#np.save('validation_Y.npy', validation_Y)
