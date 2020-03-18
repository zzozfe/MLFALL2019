import numpy as np
import pandas as pd
import sys

trainxcsv = sys.argv[3]
trainycsv = sys.argv[4]
testcsv = sys.argv[5]
anscsv = sys.argv[6]

train_X = pd.read_csv(trainxcsv)
train_Y = pd.read_csv(trainycsv,header = None)
test_X = pd.read_csv(testcsv)
train_X['fnlwgt'] = train_X['fnlwgt'].clip(0,800000)
test_X['fnlwgt'] = test_X['fnlwgt'].clip(0,800000)

def stander(x):
    for c in x.columns:
        mean = x[c].mean()
        std = x[c].std()
        if std != 0 :
            x[c] = x[c].map(lambda x : (x-mean)/std)
    return x    

def pro(x):
    if x > 0 :
        x = 1
    else:
        x = 0
    return x    

conx = pd.concat((train_X, test_X))  
conx['capital_gain'] = conx['capital_gain'].map(pro)
conx['capital_loss'] = conx['capital_loss'].map(pro)
conx = stander(conx)
train_X = conx.iloc[0:train_X.shape[0],:]
test_X = conx.iloc[train_X.shape[0]::,:]

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(train_X,train_Y)
pred = gbc.predict(test_X)

predict = pd.DataFrame()
ids = []
values = []
for i in range(test_X.shape[0]):
    ids.append(i+1)
    values.append(pred[i])
predict['id'] = ids
predict['label'] = values

predict.to_csv(anscsv,index=False)