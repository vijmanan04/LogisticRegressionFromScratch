import pandas as pd
import numpy as np
df = pd.read_csv("/content/framingham.csv")
df = df.dropna()
train = df.iloc[0:3000]
train.drop(['male', 'currentSmoker', 'BPMeds', 'cigsPerDay', 'prevalentHyp', 'diabetes'], axis = 1, inplace = True) 
test = df.iloc[3000:len(df)]
test.drop(['male', 'currentSmoker', 'BPMeds', 'cigsPerDay', 'prevalentHyp', 'diabetes'], axis = 1, inplace = True) 



y_train = train['TenYearCHD']
train.drop(['TenYearCHD'], axis = 1, inplace=True)
X_train = train
print(y_train.isnull().sum().sum())

X_train = X_train.to_numpy(dtype=np.float128)
y_train = y_train.to_numpy(dtype=np.float128)

m = len(train)
n = len(train.columns)
