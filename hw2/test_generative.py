import pandas as pd
import numpy as np
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

def normalval(x,mu,cov):
    cov_inv = pd.DataFrame(np.linalg.pinv(cov.values), cov.columns)
    vl = np.sqrt(((2*np.pi) ** x.shape[0]) * abs(np.linalg.det(cov)))
    vr1 = np.dot((x - mu), cov_inv)
    vr2 = (x - mu).to_numpy()  
    vr =  np.exp(-0.5*np.dot(vr1, vr2)) 
    value = vr*vl
    return value   

def pgm(x,y):
    x['pred'] = y.values
    pc0 = (y == 0).sum()[0]/y.shape[0]
    pc1 = 1-pc0
    mu0 = x[x['pred'] == 0].mean()
    mu0 = mu0.drop(['pred'])
    mu1 = x[x['pred'] == 1].mean()
    mu1 = mu1.drop(['pred'])
    std0 = x[x['pred'] == 0].std()
    std0 = std0.drop(['pred'])
    std1 = x[x['pred'] == 1].std()
    std1 = std1.drop(['pred']) 
    cov0 = x[x['pred'] == 0].cov().drop(['pred'],axis=1).drop(['pred'])
    cov1 = x[x['pred'] == 1].cov().drop(['pred'],axis=1).drop(['pred'])
    cov = (cov0*x[x['pred']==0].shape[0] + cov1*x[x['pred']==1].shape[0])/(x.shape[0])

    cov_inv = np.linalg.pinv(cov.values)
    w = np.dot(mu0-mu1, cov_inv)
    b = -0.5*np.dot(np.dot(mu0, cov_inv), mu0) +0.5*np.dot(np.dot(mu1, cov_inv), mu1) +         np.log(x[x['pred']==0].shape[0]/x[x['pred']==1].shape[0])
    return w,b


conx = pd.concat((train_X, test_X))  
conx['capital_gain'] = conx['capital_gain'].map(pro)
conx['capital_loss'] = conx['capital_loss'].map(pro)
conx = stander(conx)
train_X = conx.iloc[0:train_X.shape[0],:]
test_X = conx.iloc[train_X.shape[0]::,:]

w,b = pgm(train_X,train_Y)

predict = pd.DataFrame()
ids = []
values = []
z = np.dot(test_X, w) + b
pc0x = 1/(1 + np.exp(-z))
for i in range(len(pc0x)):
    ids.append(i+1)
    if pc0x[i] > 0.5:
        values.append(0)
    else:
        values.append(1)
predict['id'] = ids
predict['label'] = values

predict.to_csv(anscsv,index=False)