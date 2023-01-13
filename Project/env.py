import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm
from pyspark import SparkConf
from pyspark.sql import SparkSession
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.regularizers import L1L2
import matplotlib.pyplot as plt

def genLSTM(input_size):
    print(input_size)
    model=Sequential()
    model.add(LSTM(units=4,activation='leaky_relu',return_sequences=True,input_shape=(input_size,1)))
    # model.add(Dropout(0.1))
    model.add(LSTM(units=8,activation='leaky_relu',return_sequences=True,kernel_regularizer=L1L2(0.01,0.01)))
    # model.add(Dropout(0.1))
    model.add(LSTM(units=8,activation='leaky_relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # model.summary() # Just output info
    return model
