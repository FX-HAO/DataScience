#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Sorry, I just can't tell you where the data is, it's from my company
df = pd.read_csv('registrations20190924-24636-8a9jza.csv')

audit_state_columns = {'success': 1, 'failure': 0}
df = df.replace({'audit_state': audit_state_columns})

feature_cols = []
columns = df.columns
for i in range(len(columns)):
    cate_len = len(df[columns[i]].notnull().astype('category').cat.categories)
    if cate_len > 1:
        feature_cols.append(columns[i])

X = df[feature_cols]
y = df.audit_state

for i in range(len(feature_cols)):
    X[feature_cols[i]] = X[feature_cols[i]].notnull().astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

# Visualizing decision tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('registrations.png')
Image(graph.create_png())

import matplotlib.pyplot as plt
import seaborn as sns

# ROC curve
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
