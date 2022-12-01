import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler

def getRANGADH(start,end):
    date_start = datetime.strptime(start, '%d/%m/%Y')
    date_end = datetime.strptime(end, '%d/%m/%Y')
    v = date_end.year - date_start.year;

    if v <= 18:
        return 0
    elif 19 <= v <= 25:
        return 1
    elif 26 <= v<= 30:
        return 2
    elif 31 <= v<= 35:
        return 3
    elif 36 <= v <= 40:
        return 4
    elif 41 <= v<= 45:
        return 5
    elif 46 <= v <= 50:
        return 6
    elif 51 <= v <= 55:
        return 7
    elif v >= 56:
        return 8

def replaceNans(v):
    if pd.isna(v):
        return "ND"
    else:
        return "DV"

demissionaires = pd.read_csv('donnees_banque/table1.csv')
societaires = pd.read_csv('donnees_banque/table2.csv')

societaires.drop(societaires[societaires['DTNAIS'] == '0000-00-00' ].index, inplace = True)
societaires.drop(societaires[societaires['DTNAIS'] == '1900-01-00' ].index, inplace = True)

societaires['RANGAGEAD'] = societaires.apply(lambda x: getRANGADH(x['DTNAIS'], x['DTADH']), axis=1)

societaires = societaires.drop(['ID', 'DTNAIS', 'DTADH', 'DTDEM', 'BPADH','CDTMT'], axis=1)

demissionaires = demissionaires[societaires.columns]
demissionaires = demissionaires.dropna()

tmp = []
for r in demissionaires['RANGAGEAD']:
    tmp.append(int(r.split(' ')[0]))
demissionaires['RANGAGEAD'] = tmp

societaires = pd.concat([societaires,demissionaires],axis=0)

societaires['CDMOTDEM'] = societaires['CDMOTDEM'].apply(lambda x: replaceNans(x))

numerical = societaires[['MTREV','NBENF']]
categorical = societaires[['CDSEXE','CDSITFAM','CDCATCL']]

normalizer = Normalizer()
numerical = normalizer.fit_transform(numerical)
numerical = pd.DataFrame(numerical, columns=['MTREV','NBENF'])

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(categorical)
categorical_one_hot = enc.transform(categorical).toarray()

feature_names = enc.get_feature_names_out()

X_cat = pd.DataFrame(categorical_one_hot,columns=feature_names).astype(int)
X_cat_fuzed = pd.merge(X_cat,numerical,left_index=True,right_index=True)

y = societaires['CDMOTDEM']

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

#https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
x_train, x_test, y_train, y_test = train_test_split(X_cat_fuzed, y, test_size=1 - train_ratio)


x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_score

clf = LogisticRegression(random_state=0, max_iter=10)


print(x_train.shape)
print(clf.fit(x_train, y_train))
print(clf.score(x_val, y_val))
y = clf.predict_proba(x_val)
y_pred = []
for val in y:
    y_pred.append(np.max(val))

for i in range(len(clf.coef_[0])):
    if np.abs(clf.coef_[0][i]) > 1:
        print(i, ', ', x_train.columns[i], ' :', clf.coef_[0][i])

y_val = y_val.replace(['ND'], 0)
y_val = y_val.replace(['DV'], 1)


from sklearn.metrics import det_curve
from sklearn.metrics import det_curve, DetCurveDisplay

fpr, fnr, thresholds = det_curve(y_val, y_pred)
display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="LogisticRegression")

display.plot()
# plt.show()