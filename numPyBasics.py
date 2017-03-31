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
    
    






