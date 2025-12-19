import os
import sys
import pickle
import shap

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import numpy as np

sys.path.append('./')
sys.path.append('../scripts/')
import utils

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

seed = 42

DATA_DIR = ".../jgr-manuscript/"

file_data = DATA_DIR + "data/data_train_test.pickle"  # path to the preprocessed data
data_train, data_test, y_train_full, y_train_cat_full, X_train_full, y_test, y_test_cat, X_test = utils.load_data(file_data)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=seed) # additional train-test-split for validation data inside the full trining set

sc = MinMaxScaler((0,1))
# sc = StandardScaler
X_train_sc = sc.fit_transform(X_train)
X_val_sc = sc.transform(X_val)
X_test_sc = sc.transform(X_test)

y_train_sc = y_train.to_numpy().reshape(-1, 1)
y_val_sc = y_val.to_numpy().reshape(-1, 1)
y_test_sc = y_test.to_numpy().reshape(-1, 1)

def build_model(input_dim,
                output_dim,
                n_units1=512,
                n_units2=256,
                n_units3=256,
                activation='relu',
                dropout_rate=0.1,
                learning_rate=2e-2,
                epochs=1000,
                patience=50):
    # Define layers
    model = Sequential([
        Dense(n_units1, input_dim=input_dim, activation=activation, kernel_initializer='normal'),
        Dropout(dropout_rate),
        Dense(n_units2, activation=activation, kernel_initializer='normal'),
        Dropout(dropout_rate),
        Dense(n_units3, activation=activation, kernel_initializer='normal'),
        Dropout(dropout_rate),
        Dense(output_dim, activation='linear', kernel_initializer='normal')  # regression head
    ])
    epochs = 1000
    patience = 50 #32
    opt = SGD(learning_rate=learning_rate, momentum=0.8)
    mse = MeanSquaredError()
    

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=[mse])
    return model

def get_keras_regressor(input_dim, output_dim=1):
    """Create a KerasRegressor wrapper compatible with GridSearchCV."""
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.75,
        patience=20,
        min_lr=1e-4,
        verbose=1
    )
    
    regressor = KerasRegressor(
        model=build_model,
        input_dim=input_dim,
        output_dim=output_dim,
        # Default hyperparameters (can be overridden by GridSearchCV)
        n_units1=512,
        n_units2=256,
        n_units3=256,
        activation='relu',
        dropout_rate=0.1,
        learning_rate=2e-2,
        # Training parameters
        epochs=1000,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return regressor

input_dim = X_train_sc.shape[1] # Number of input features
output_dim = y_train_sc.shape[1] 

# Create the regressor
regressor = get_keras_regressor(input_dim=input_dim,output_dim=output_dim)

# Define parameter grid
param_grid = {
    'epochs': [250], # use 2 for testing, 500-1000 for real training, takes time so 250?
    'n_units1': [256, 512],
    'n_units2': [256, 512],
    #'n_units3': [64, 128, 256],
    'n_units3': [128, 256],
    #'dropout_rate': [0.05, 0.1, 0.2],
    'dropout_rate': [0.1],
    'learning_rate': [1e-1],
    'batch_size': [256, 512],
    #'activation' : ['relu', 'gelu','selu'],
    'activation' : ['relu', 'gelu']
}

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=regressor,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error', # use whatever you have used for the other cases
    verbose=2,
    n_jobs=1,  # Use 1 for GPU; increase for CPU-only
    refit=True
)

# Fit
grid_search.fit(X_train_sc, y_train_sc, validation_data=(X_val_sc,y_val_sc), verbose=0)

# Results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Access best model
best_model = grid_search.best_estimator_
# Save the best model
best_model.model_.save('models/best_ffnn_regression_model.keras') # previously it was .h5

