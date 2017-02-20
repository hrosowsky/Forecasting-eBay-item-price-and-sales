# -*- coding: utf-8 -*-
"""
Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')



def linear_regression(series):
    name = series.name
    # Create features and predicition variable.
    features = pd.concat([
                        percent_change_lead(series, 7) # Predicition variable
                        , percent_change(series,1) # Feature 1
                        , percent_change(series,2) # Feature 2
                        , percent_change(series,7) # Feature 3
                        , percent_change(series,14)# Feature 4
                        ], axis=1).dropna(axis=0)
    

    
    # Construct training set to fit model
    train_X = features.iloc[:100, 1:]
    train_Y = features.iloc[:100, 0].rename('train_y')
    train_original = series.reindex(train_Y.index).rename(name)
    
    # Construct testing set to evaulate the models performance out of sample
    test_X = features.iloc[100:, 1:]
    test_Y = features.iloc[100:, 0].rename('test_y')
    test_original = series.reindex(test_Y.index).rename(name)
    
    # Load regression model
    from sklearn.linear_model import LinearRegression
    
    # Learn linear regression weights
    model = LinearRegression().fit(train_X.values, train_Y.values)
    
    # Run in and out of sampling testing
    train_Yp = model.predict(train_X)
    test_Yp = model.predict(test_X)
    #R-squared error   
    r2 = model.score(train_X, train_Y)
    # Merge everything into a single pandas dataframe
    train_Yp = pd.Series(train_Yp, index=train_Y.index, name='train_yp')
    test_Yp = pd.Series(test_Yp, index=test_Y.index, name='test_yp')
    
    return pd.concat([train_original, train_Y, train_Yp], axis=1), pd.concat([test_original, test_Y, test_Yp],axis=1)

def percent_change(series, lag):
    return (series/series.shift(lag) - 1.).rename(series.name + '_lag_' + str(lag))
    
def percent_change_lead(series, lead):
    return (series.shift(-lead)/series - 1.).rename(series.name + '_lead_' + str(lead))
    
def mean_squared_error(series_y, series_yp):
    return ((series_y-series_yp)**2).mean()    
    
def plot(product, ptrain, ptest, strain, stest):

    plt.figure()
    plt.suptitle(product)

    # Price
    plt.subplot(221)
    plt.title('7 day avg price change forecast', fontsize=9)
    
    ptrain[['train_y', 'train_yp']].plot(ax=plt.gca())
    ptest[['test_y', 'test_yp']].plot(ax=plt.gca())
    
    plt.gca().legend(loc='upper left', markerscale=0.5,fontsize=7)
    
    plt.subplot(222)
    plt.title('7 day avg price forecast', fontsize=9) # Plot the price forcast p(t+7) = p(t)*(1+y(t))
    
    predict_train = ((ptrain['train_yp']+1.)*ptrain['avg_price']).rename('train_yp')
    pd.concat([ptrain['avg_price'].shift(-7), predict_train], axis=1).dropna().plot(ax=plt.gca())
    
    predict_test = ((ptest['test_yp']+1.)*ptest['avg_price']).rename('test_yp')
    pd.concat([ptest['avg_price'].shift(-7), predict_test], axis=1).dropna().plot(ax=plt.gca())
    
    plt.gca().legend(loc='upper left', markerscale=0.5,fontsize=7)
    
    # Sales
    plt.subplot(223)
    plt.title('7 day sales change forecast', fontsize=9)
    
    strain[['train_y', 'train_yp']].plot(ax=plt.gca())
    stest[['test_y', 'test_yp']].plot(ax=plt.gca())
    
    plt.gca().legend(loc='upper left', markerscale=0.5,fontsize=7)
    
    plt.subplot(224)
    plt.title('7 day sales forecast', fontsize=9) # Plot the price forcast p(t+7) = p(t)*(1+y(t))
    
    predict_train = ((strain['train_yp']+1.)*strain['n_sold']).rename('train_yp')
    pd.concat([strain['n_sold'].shift(-7), predict_train], axis=1).dropna().plot(ax=plt.gca())
    
    predict_test = ((stest['test_yp']+1.)*stest['n_sold']).rename('test_yp')
    pd.concat([stest['n_sold'].shift(-7), predict_test], axis=1).dropna().plot(ax=plt.gca())
    
    
    plt.gca().legend(loc='upper left', markerscale=0.5,fontsize=7)
    
    plt.tight_layout()
    