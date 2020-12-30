# Installing libraries and dependencies
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pickle

def preprocess_data_for_request(data2):

    data2.vendor = mapping_vendor[data2.vendor[0]]
    data2.description = mapping_description[data2.description[0]]
    data2.category = mapping_category[data2.category[0]]
    data2.Location = mapping_Location[data2.Location[0]]
    
    return data2

def predict_amount(config, model):

    processed_data = preprocess_data_for_request(config)
    print(processed_data)
    y_pred2 = model.predict(processed_data)
    return y_pred2