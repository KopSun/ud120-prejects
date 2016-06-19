#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score

#clf = tree.DecisionTreeClassifier()
#clf.fit(features, labels)
#pred = clf.predict(features)
#
#acc = accuracy_score(labels, pred)
#print acc

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .3, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

for i in range(len(pred)):
    if pred[i] == 1:
        pred[i] = 0


acc = accuracy_score(labels_test, pred)
print acc

#poi_pred_cnt = 0
#for i in pred:
#    if i == 1:
#        poi_pred_cnt += 1

#print poi_pred_cnt
#print len(pred)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

ps = precision_score(labels_test, pred)
print ps
rs = recall_score(labels_test, pred)
print rs