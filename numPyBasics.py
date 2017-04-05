import pandas as pd

df = pd.read_csv('GBPUSD4h.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%-m/%-d/%y %-H:%M')
bbDf = df[['Date', 'High', 'Low', 'Close']].copy()

import talib
period = 20
nbdevup = 2.0
nbdevdn = 2.0

closes = bbDf['Close'].values
upper, middle, lower = talib.BBANDS(closes, timeperiod=20, matype=0) 
bbDf['BB_Upper'] = upper
bbDf['BB_Middle'] = middle
bbDf['BB_Lower'] = lower

%matplotlib inline

from matplotlib import pyplot as plt

bbDf.plot(x='Date', y=['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower'])
plt.show()

bbDf['BB_PctB'] = 100 * (bbDf['Close'] - bbDf['BB_Lower'])/(bbDf['BB_Upper'] - bbDf['BB_Lower'])
bbDf['pc_Upper'] = bbDf['BB_Upper'].pct_change()
bbDf['pc_Middle'] = bbDf['BB_Middle'].pct_change()
bbDf['pc_Lower'] = bbDf['BB_Lower'].pct_change()
bbDf['pc_PctB'] = bbDf['BB_PctB'].pct_change()
bbDf['Return'] = bbDf['Close'].pct_change()
bbDf['Class'] = bbDf['Return'].map(lambda x: 'Up' if x>=0 else 'Down')

bbDf['ClassShifted'] = bbDf['Class'].shift(-1)

modelDf = bbDf.loc[20:][['BB_PctB', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'pc_Upper', 'pc_Middle', 'pc_Lower', 'pc_PctB', 'ClassShifted']] .copy()
modelDf['ClassShifted'] = modelDf['ClassShifted'].map(lambda x: 1 if x == 'Up' else -1)
modelDf.head()

import random
random.seed(1)

X = modelDf[['BB_PctB', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'pc_Upper', 'pc_Middle', 'pc_Lower', 'pc_PctB']].values
Y = modelDf['ClassShifted'].values
featureNames = ['BB_PctB', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'pc_Upper', 'pc_Middle', 'pc_Lower', 'pc_PctB']

rf = RandomForestRegressor()
rf.fit(X, Y)

sorted(zip(rf.feature_importances_, featureNames), reverse=True)

from collections import defaultdict

scores = defaultdict(list)

#testRatio = 0.3, len(X)=506, so trainSz=(506*0.7)  testSz=152
#Repeat 100 times
count=0
for train_idx, test_idx in ShuffleSplit(len(X), 100, 0.2):
    #if count == 1:
    #    break
        
    #print("{} {}".format(train_idx.shape, test_idx.shape))
    X_train, X_test = X[train_idx],  X[test_idx]
    Y_train, Y_test = Y[train_idx],  Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    
    #X.shape[1] = 13, which is all the columns(features)
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[featureNames[i]].append((acc-shuff_acc)/acc)
        if count == 99: 
            print("{} {}".format(featureNames[i], len(scores[featureNames[i]]) ))
        
    count += 1

    sorted([(np.mean(score), feat) for feat, score in scores.items()], reverse=True)
    
    


import pandas as pd
import numpy as np
from pandas_datareader import data
from math import sqrt


#make sure the NYSE.txt file is in the same folder as your python script file
stocks = pd.read_csv('NYSE.txt',delimiter="\t")
 
#set up our empty list to hold the stock tickers
stocks_list = []
 
#iterate through the pandas dataframe of tickers and append them to our empty list
for symbol in stocks['Symbol']:
    stocks_list.append(symbol)
    
len(stocks_list)

stocks_list[:5]

#create empty list to hold our return series DataFrame for each stock
frames = []
 
for stock in stocks_list:
    try:
        #download stock data and place in DataFrame
        df = data.DataReader(stock, 'yahoo',start='1/1/2000')
        
        df['Stdev'] = df['Close'].rolling(window=90).std()
        
        #create a column to hold our 20 day moving average
        df['Moving Average'] = df['Close'].rolling(window=20).mean()
        
        #create a column which holds a TRUE value if the gap down from previous day's low to next 
        #day's open is larger than the 90 day rolling standard deviation
        df['Criteria1'] = (df['Open'] - df['Low'].shift(1)) < -df['Stdev'] 
        
        #create a column that holds a TRUE value if both above criteria are also TRUE
        df['BUY'] = df['Criteria1'] & df['Criteria2']
        
        #calculate daily % return series for stock
        df['Pct Change'] = (df['Close'] - df['Open']) / df['Open']
        
        #create a strategy return series by using the daily stock returns where the trade criteria above are met
        df['Rets'] = df['Pct Change'][df['BUY'] == True] 
        
        #append the strategy return series to our list
        frames.append(df['Rets'])
        
        except:
            pass
        
       #concatenate the individual DataFrames held in our list- and do it along the column axis
masterFrame = pd.concat(frames,axis=1)
 
#create a column to hold the sum of all the individual daily strategy returns
masterFrame['Total'] = masterFrame.sum(axis=1)
 
#create a column that hold the count of the number of stocks that were traded each day
#we minus one from it so that we dont count the "Total" column we added as a trade.
masterFrame['Count'] = masterFrame.count(axis=1) - 1
 
#create a column that divides the "total" strategy return each day by the number of stocks traded that day to get equally weighted return.
masterFrame['Return'] = masterFrame['Total'] / masterFrame['Count']


masterFrame['Return'].dropna().cumsum().plot()

masterFrame['Return'].dropna().cumsum().plot()
 




import pandas as pd
import talib
%matplotlib inline

from matplotlib import pyplot as plt

#period = 20
#nbdevup = 2.0
#nbdevdn = 2.0

def getBB(bbDf, period=20, nbdevup=2.0, nbdevdn=2.0):
    closes = bbDf['Close'].values
    upper, middle, lower = talib.BBANDS(closes, timeperiod=20, matype=0)
    bbDf['BB_Upper'] = upper
    bbDf['BB_Middle'] = middle
    bbDf['BB_Lower'] = lower
    bbDf.plot(x='Date', y=['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower'])
    plt.show()
    
    return bbDf
    

    
#n is also the shift upward period in pd.series

def generateModelDf(bbDf, n=1):
    bbDf['BB_PctB'] = 100 * (bbDf['Close'] - bbDf['BB_Lower'])/(bbDf['BB_Upper'] - bbDf['BB_Lower'])
    bbDf['pc_Upper'] = bbDf['BB_Upper'].pct_change(n)
    bbDf['pc_Middle'] = bbDf['BB_Middle'].pct_change(n)
    bbDf['pc_Lower'] = bbDf['BB_Lower'].pct_change(n)
    bbDf['pc_PctB'] = bbDf['BB_PctB'].pct_change(n)
    bbDf['Return'] = bbDf['Close'].pct_change(n)
    bbDf['Class'] = bbDf['Return'].map(lambda x: 'Up' if x>=0 else 'Down')
    bbDf['ClassShifted'] = bbDf['Class'].shift(n*(-1))
    cutSz = 20+ (n-1)
    modelDf = bbDf.loc[cutSz:][['BB_PctB', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'pc_Upper', 'pc_Middle', 'pc_Lower', 'pc_PctB', 'Return', 'ClassShifted']] .copy()
    #modelDf = bbDf[['BB_PctB', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'pc_Upper', 'pc_Middle', 'pc_Lower', 'pc_PctB', 'ClassShifted']] .copy()
    modelDf['ClassShifted'] = modelDf['ClassShifted'].map(lambda x: 1 if x == 'Up' else -1)
        
    return modelDf    


from sklearn.ensemble import RandomForestRegressor
import random

random.seed(100)


#Get RF "mean decrase impurity"
def getRfMeanDecreaseImpurity(rf, modelDf):
    X = modelDf[['BB_PctB', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'pc_Upper', 'pc_Middle', 'pc_Lower', 'pc_PctB']].values
    Y = modelDf['ClassShifted'].values
    featureNames = ['BB_PctB', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'pc_Upper', 'pc_Middle', 'pc_Lower', 'pc_PctB']
    
    #import numpy as np
    #print(np.isnan(X).any())
    #print(np.isinf(Y).any()

    #rf = RandomForestRegressor()
    rf.fit(X, Y)
    
    print("Features sorted by their score:")
    print(sorted(zip(rf.feature_importances_, featureNames), reverse=True))
    
    return X, Y
    
    
df = pd.read_csv('IF_10.csv')
df['Date'] = pd.to_datetime((df['Date'] + ' ' + df['Time']), format='%Y/%m/%d %H:%M:%S')
del df['Time']

bbDf = df[['Date', 'High', 'Low', 'Close']].copy()
bbDf = getBB(bbDf)
   
bbDf.head(30)

modelDf = generateModelDf(bbDf)

modelDf.head(30)

rf = RandomForestRegressor()
X, Y = getRfMeanDecreaseImpurity(rf, modelDf)

    

    




