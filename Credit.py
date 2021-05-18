import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
import sys

data = pd.read_csv('creditcard.csv')
#print(data.shape)
data = data.sample(frac=0.2,random_state=1)
#print(data.shape)

data.hist(figsize=(15,15))
#plt.show()

fraud = data[data['Class']==1]
valid = data[data['Class']==0]

outlier = len(fraud)/float(len(valid))
#print(outlier)

#print('Fraud cases: {}'.format(len(fraud)))
#print('Valid cases: {}'.format(len(valid)))

fig = plt.figure(figsize=(12,9))
sns.heatmap(data.corr(), vmax=.8, square= True)
#plt.show()

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['Class']]

target = 'Class'

X=data[columns]
Y=data[target]

print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state = 1
classifiers = {
    'Isolation Forest': IsolationForest(max_samples= len(X),contamination=outlier,random_state= state),
    'Local Outlier Factor': LocalOutlierFactor(
        n_neighbors= 20,
        contamination=outlier
    )
}

n_outliers = len(fraud)

for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

y_pred[y_pred ==1] = 0
y_pred[y_pred ==-1] = 1

n_errors = (y_pred != Y).sum()

print('{}: {}'.format(clf_name, n_errors))
print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))