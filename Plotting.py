# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:22:24 2020

@author: Gianluca
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

baseline = np.load('best_baseline_prediction.npy')
Conv = np.load('Best_Conv_Dense_prediction.npy')
FullyConv = np.load('Best_FullyConv_prediction.npy')
true = np.load('validation_Y.npy')

X = np.arange(84, 92, 1)

fig = plt.figure(figsize = (19.2, 10.8))
ax = fig.add_subplot(1, 1, 1)


# ax.plot(X, true[7], ls = '-', marker = 'o', label = 'True Values')
# ax.plot(X, baseline[7], ls = '-', marker = 'x', label = 'Baseline Model')
# ax.plot(X, Conv[7], ls = '-', marker = 'v', label = 'Convolutional-Dense')
# ax.plot(X, FullyConv[7], ls = '-', marker = '^', label = 'Fully Convolutional')
# ax.legend(fontsize = 'x-large')
# ax.set_xlabel('Day', fontsize = 'xx-large')
# ax.set_ylabel('Total Cases', fontsize = 'xx-large')
# ax.set_title('8 Days Prediction', fontsize = 'xx-large')
# plt.tight_layout()
# plt.show()
# plt.savefig('8_Days_prediction.png', dpi = 200)
#plt.close()

Conv_Df = pd.read_csv('Conv_Dense_4D.csv')
baseline_df = pd.read_csv('Baseline_4D.csv')
FullyConv_df = pd.read_csv('FullyConv_4D.csv')

ax.plot(Conv_Df['epoch'], Conv_Df['val_loss'], label = 'Convolutional - Dense')
ax.plot(baseline_df['epoch'], baseline_df['val_loss'], label = 'Baseline')
ax.plot(FullyConv_df['epoch'], FullyConv_df['val_loss'], label = 'Fully Convolutional')
ax.set_ylabel('Mean Squared Error (log)', fontsize = 'xx-large')
ax.set_xlabel('Epoch', fontsize = 'xx-large')
ax.set_title('Training 4 Days', fontsize = 24)
ax.legend(fontsize = 'xx-large')
plt.tight_layout()
plt.yscale('log')
plt.show()
plt.savefig('Training_4Days.png', dpi = 200)
