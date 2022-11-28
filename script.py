import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler

demissionaires = pd.read_csv('donnees_banque/table1.csv')
societaires = pd.read_csv('donnees_banque/table2.csv')

#print(societaires.corr()) correlation entre les variables
#print(societaires.cov()) covariance




#societaires['CDMOTDEM'].fillna('ND', inplace = True)



demissionaires.drop(['AGEAD','AGEDEM','CDTMT','CDDEM','ADH','ANNEEDEM'],axis=1,inplace=True)
societaires.drop(['BPADH','CDTMT'],axis=1,inplace=True)
#print(societaires.dtypes)
#print(demissionaires['RANGADH'].unique())
#['7  30-34' '6  25-29' '5  20-24' '4  15-19' '3  10-14' '2  5-9' '1  1-4'
# nan]
def getRANGADHfromnum(v):
    if v <= 4:
        return 1
    elif v<= 9:
        return 2
    elif v<= 10:
        return 3
    elif v<= 15:
        return 4
    elif v<= 20:
        return 5
    elif v<= 25:
        return 6
    elif v<= 30:
        return 7
    elif v>30:
        return 8


staple_date = datetime.strptime('30/12/2007','%d/%m/%Y')

def getRANGfromDate(date):
    date_conv = datetime.strptime(date,'%d/%m/%Y')
    if (date_conv.year == 1900):
        return 0
    else:
        return getRANGADHfromnum(0)
    
def getRANGADH(start,end):
    
    date_start = datetime.strptime(start, '%d/%m/%Y')
    date_end = datetime.strptime(end, '%d/%m/%Y')
    if(date_end.year == 1900):
        return getRANGADHfromnum(staple_date.year - date_start.year)
    else:
        return getRANGADHfromnum(date_end.year - date_start.year)

def replaceNans(v):
    if pd.isna(v):
        return "ND"
    else:
        return "DV"

#print(societaires)
societaires['RANGADH'] = societaires.apply(lambda x: getRANGADH(x['DTADH'],x['DTDEM']),axis=1)
#societaires['RANGAGEDEM'] = societaires.apply(lambda x: getRANGfromDate(x['DTDEM']))
societaires.drop(societaires[societaires['DTNAIS'] == '0000-00-00' ].index, inplace = True)
societaires.drop(societaires[societaires['DTNAIS'] == '1900-01-00' ].index, inplace = True)

age_adh = []
for column, row in societaires[['DTADH', 'DTDEM', 'DTNAIS']].iterrows():
    dt_nais = datetime.strptime(row['DTNAIS'], '%d/%m/%Y')
    dt_adh = datetime.strptime(row['DTADH'], '%d/%m/%Y')
    age_adh.append(getRANGADHfromnum(round(max(0, (dt_adh-dt_nais).days/365))))
societaires = societaires.drop(['ID', 'DTNAIS', 'DTADH', 'DTDEM'], axis=1)

societaires['RANGADH'] = age_adh

demissionaires = demissionaires[societaires.columns]
demissionaires = demissionaires.dropna()

tmp = []
for r in demissionaires['RANGADH']:
    tmp.append(int(r.split(' ')[0]))
demissionaires['RANGADH'] = tmp

print('table 1 : \n', demissionaires.head())
print('table 2 : \n', societaires.head())

print(demissionaires.dtypes)
print(societaires.dtypes)


societaires = pd.concat([societaires,demissionaires],axis=0)
#societaires = societaires.merge(demissionaires,how='outer')
societaires['CDMOTDEM'] = societaires['CDMOTDEM'].apply(lambda x: replaceNans(x))
#ND if is nan in societaires
print(societaires)

numerical = societaires[['MTREV','NBENF']]
categorical = societaires[['CDSEXE','CDSITFAM','CDCATCL','RANGADH']]

normalizer = Normalizer()
numerical = normalizer.fit_transform(numerical)
numerical = pd.DataFrame(numerical, columns=['MTREV','NBENF'])

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(categorical)
categorical_one_hot = enc.transform(categorical).toarray()
#print(categorical_one_hot)
feature_names = enc.get_feature_names_out()

X_cat = pd.DataFrame(categorical_one_hot,columns=feature_names).astype(int)
X_cat_fuzed = pd.merge(X_cat,numerical,left_index=True,right_index=True)
#tant que l'ordre des index est inchange ce merge permet d'unifier les valeurs categorique et numerique
#apres encodage/pretraitement
print(X_cat_fuzed)
print(X_cat_fuzed.dtypes)

y = societaires['CDMOTDEM']

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

#https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
x_train, x_test, y_train, y_test = train_test_split(X_cat_fuzed, y, test_size=1 - train_ratio)


x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 




#print(x_train, x_val, x_test)
####machine learning time

#print(demissionaires.head())
#print(societaires.head())
#print(demissionaires.columns)
#print("nombre de valeurs unique RANGAGEAD table1:"+demissionaires['RANGAGEAD'].unique())
#print("nombre de valuers unique RANGAGEDEM table1:"+demissionaires['RANGAGEDEM'].unique())
# 'nombre de valeurs RANGAGEAD table1:1  19-25'
# 'nombre de valeurs RANGAGEAD table1:2  26-30'
#['nombre de valeurs RANGAGEAD table1:3  31-35'
# 'nombre de valeurs RANGAGEAD table1:4  36-40'
# 'nombre de valeurs RANGAGEAD table1:5  41-45'
# 'nombre de valeurs RANGAGEAD table1:6  46-50']
# 'nombre de valeurs RANGAGEAD table1:7  51-55'
# 'nombre de valeurs RANGAGEAD table1:8  56-+'

# 'nombre de valuers unique RANGAGEDEM table1:b  71-+'
# 'nombre de valuers unique RANGAGEDEM table1:a  66-70'
 #'nombre de valuers unique RANGAGEDEM table1:8  56-60'
 #'nombre de valuers unique RANGAGEDEM table1:7  51-55'
# 'nombre de valuers unique RANGAGEDEM table1:6  46-50'
# 'nombre de valuers unique RANGAGEDEM table1:5  41-45'
# 'nombre de valuers unique RANGAGEDEM table1:3  31-35'
# 'nombre de valuers unique RANGAGEDEM table1:4  36-40'
# 'nombre de valuers unique RANGAGEDEM table1:2  26-30'
 #'nombre de valuers unique RANGAGEDEM table1:1  19-25']
 
#age d'adhésion par catégorie de demission
#fig = plt.figure()
#demissionaires.plot.hist(column=['AGEAD'],by="CDMOTDEM",figsize=(10,8))
#plt.savefig('fig/AGEAD_by_CDMOTDEM_histogram')
#plt.close(fig)

#age d'adhésion par situation familial
#fig = plt.figure()
#demissionaires.plot.hist(column=['AGEAD'],by="CDSITFAM",figsize=(20,16))
#plt.savefig('fig/AGEAD_by_CDSITFAM_histogram')
#plt.close(fig)

#motif de demission par categorie de client
#fig = plt.figure()
#demissionaires.plot.hist(column=['CDCATCL'],by="CDMOTDEM",figsize=(20,16))
#plt.savefig('fig/CDMOTDEM_by_CDCATCL_histogram')
#plt.close(fig)


#pie plot des motifs de demission



