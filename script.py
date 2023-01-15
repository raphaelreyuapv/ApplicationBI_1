import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler
from joblib import dump,load
from sklearn.inspection import permutation_importance

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
#societaires = societaires.merge(demissionaires,how='outer')
societaires['CDMOTDEM'] = societaires['CDMOTDEM'].apply(lambda x: replaceNans(x))
#ND if is nan in societaires
print(societaires)

numerical = societaires[['MTREV','NBENF']]
categorical = societaires[['CDSEXE','CDSITFAM','CDCATCL','RANGAGEAD']]

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
#tant que l'ordre des index est inchange ce merge permet d'unifier les valeurs categorique et numerique dans le meme dataframe
#apres encodage/pretraitement
print(X_cat_fuzed)
print(X_cat_fuzed.dtypes)

y = societaires['CDMOTDEM']
print(societaires['CDMOTDEM'].value_counts())
####creation d'un dataset oversampled pour résoudre le désiquilibre des classes
max_size = societaires['CDMOTDEM']
societaires_ND = societaires.loc[societaires['CDMOTDEM'] == "ND"]
societaires_oversample = societaires_ND.sample(14539,replace=True)
societaires_oversample = pd.concat([societaires,societaires_oversample])
print(societaires_oversample)
categorical_oversampled = societaires_oversample[['CDSEXE','CDSITFAM','CDCATCL','RANGAGEAD']]
numerical_oversampled = societaires_oversample[['MTREV','NBENF']]
categorical_one_hot_oversampled = enc.transform(categorical_oversampled).toarray()
X_cat_oversampled = pd.DataFrame(categorical_one_hot_oversampled,columns=feature_names).astype(int)
numerical_oversampled = normalizer.transform(numerical_oversampled)
numerical_oversampled = pd.DataFrame(numerical_oversampled,columns=['MTREV','NBENF'])

X_cat_fuzed_oversampled = pd.merge(X_cat_oversampled,numerical_oversampled,left_index=True,right_index=True)
print(X_cat_fuzed_oversampled)
y_oversampled = societaires_oversample['CDMOTDEM']
train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

##Faire un dummy classifier pour tester la veracite du SVM
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import PredefinedSplit
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
#https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
#creation des dataset pour les classifieurs
x_train, x_test, y_train, y_test = train_test_split(X_cat_fuzed, y, test_size=1 - train_ratio,random_state=42)
x_train_oversampled,x_test_oversampled,y_train_oversample,y_test_oversampled = train_test_split(X_cat_fuzed_oversampled, y_oversampled, test_size=1 - train_ratio,random_state=42)
#on ne retient l'oversampling que pour le train, le jeu de test pour l'evaluation final reste le meme pour tous
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio),random_state=48) 
x_val_oversampled,x_test_oversampled,y_val_oversampled,y_test_oversampled = train_test_split(x_test_oversampled, y_test_oversampled, test_size=test_ratio/(test_ratio + validation_ratio),random_state=48) 
#creation et fit des classifieurs
#svc_clf_linear = SVC(class_weight='balanced',kernel="linear")
#svc_clf_poly = SVC(class_weight='balanced',kernel="poly")
svc_clf_rbf = SVC(class_weight='balanced',kernel="rbf",C=100,gamma=0.01)
#svc_clf_sig = SVC(class_weight='balanced',kernel="sigmoid")
#svc_clf_linear.fit(x_train,y_train)
#svc_clf_poly.fit(x_train,y_train)
svc_clf_rbf.fit(x_train,y_train)
#svc_clf_sig.fit(x_train,y_train)
dummycl = DummyClassifier(strategy="most_frequent")
dummycl.fit(x_train,y_train)
neigh_clf = KNeighborsClassifier(n_neighbors=5)
neigh_clf.fit(x_train_oversampled, y_train_oversample)
#ps = PredefinedSplit()

#verification
scores_dummy_val = cross_val_score(dummycl,x_val,y_val,cv=5)
print("Accuracy of dummy(most frequent) classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores_dummy_val.mean(), scores_dummy_val.std() * 2))

#scores_svc_val = cross_val_score(svc_clf_linear,x_val,y_val,cv=5)
#print("Accuracy of SVC classifier(linear kernel) on cross-validation: %0.2f (+/- %0.2f)" % (scores_svc_val.mean(), scores_svc_val.std() * 2))
#scores_svc_val = cross_val_score(svc_clf_poly,x_val,y_val,cv=5)
#print("Accuracy of SVC classifier(poly kernel) on cross-validation: %0.2f (+/- %0.2f)" % (scores_svc_val.mean(), scores_svc_val.std() * 2))
#scores_svc_val = cross_val_score(svc_clf_rbf,x_val,y_val,cv=5)
#print("Accuracy of SVC classifier(rbf kernel) on cross-validation: %0.2f (+/- %0.2f)" % (scores_svc_val.mean(), scores_svc_val.std() * 2))
#scores_svc_val = cross_val_score(svc_clf_sig,x_val,y_val,cv=5)
#print("Accuracy of SVC classifier(sig kernel) on cross-validation: %0.2f (+/- %0.2f)" % (scores_svc_val.mean(), scores_svc_val.std() * 2))

from sklearn.model_selection import GridSearchCV

#param_grid = {'C': [0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}
param_grid = {'n_neighbors': [2,3,5,7,9], 'weights': ['distance'], 'metric':['minkowski']}
grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=3,scoring='accuracy')
grid.fit(x_train,y_train)
print("GridSearch best params")
#print(grid.best_params_)
#print(svc_clf.coef_)
#print(svc_clf.feature_names_in_)
#mapping = dict(zip(svc_clf_linear.feature_names_in_,svc_clf_linear.coef_[0]))
#print(mapping)
#for key,value in mapping.items():
#    print("Attribue:",key," Valeur:",value)
#scores_knn_val = cross_val_score(neigh_clf,x_val,y_val,cv=2)
#print("Accuracy of K nearest neigh classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores_knn_val.mean(), scores_knn_val.std() * 2))
#print("KNN effective metric")
#print(neigh_clf.effective_metric_)
#print("Fonction de calcul des distances:Minkowski")
#C=100, gamma=0.01, kernel=rbf;

##VALIDATION
##TEST
print("------VALIDATION-------")
print("Classification report for dummy classifier")
y_pred = dummycl.predict(x_val)
print(metrics.classification_report(y_val,y_pred))
scores_knn_test = neigh_clf.score(x_val,y_val)
y_pred = neigh_clf.predict(x_val)
print("Classification report for knn:")
print(metrics.classification_report(y_val,y_val))
y_pred = svc_clf_rbf.predict(x_val)
print("Classification report for SVC:")
print(metrics.classification_report(y_val,y_pred))
print("------TEST-------")
##TEST
scores_dummy_test = dummycl.score(x_test,y_test)
print("Classification report for dummy classifier")
y_pred = dummycl.predict(x_test)
print(metrics.classification_report(y_test,y_pred))
print("Accuracy of dummy(most frequent) classifier on Test set: %0.2f" % (scores_dummy_test))
scores_knn_test = neigh_clf.score(x_test,y_test)
y_pred = neigh_clf.predict(x_test)
print("Classification report for knn:")
print(metrics.classification_report(y_test,y_pred))
print("Accuracy of K nearest neigh classifier on Test set: %0.2f" % (scores_knn_test))
scores_svc_test = svc_clf_rbf.score(x_test,y_test)
print("Accuracy of SVC classifier(rbf kernel) on Test set: %0.2f" % (scores_svc_test))
y_pred = svc_clf_rbf.predict(x_test)
print("Classification report for SVC:")
print(metrics.classification_report(y_test,y_pred))
print("-----FEATURE ANALYSIS------")
print("Feature importance for knn:")
result = permutation_importance(neigh_clf,x_test,y_test,n_repeats=1,random_state=0)
features_names_cc = x_test.columns
feature_importance = pd.DataFrame(result.importances, columns=['Importance'])
feature_importance['Feature'] = features_names_cc
print(feature_importance)
print("Feature importance for SVC:")
result = permutation_importance(svc_clf_rbf,x_test,y_test,n_repeats=1,random_state=0)
features_names_cc = x_test.columns
feature_importance = pd.DataFrame(result.importances, columns=['Importance'])
feature_importance['Feature'] = features_names_cc
print(feature_importance)
#Sauvegarde des classifieurs
dump(svc_clf_rbf,"svc_clf.joblib")
dump(dummycl,"dummycl.joblib")
dump(neigh_clf,"neigh_clf.joblib")
#Bayes=Categorical obliger
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



