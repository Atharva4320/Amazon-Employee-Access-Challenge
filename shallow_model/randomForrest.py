
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#Import svm model
from sklearn import svm
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics
from access import repeat_action_zero_columns
tr_data = pandas.read_csv("../train.csv")
tr_data = repeat_action_zero_columns(tr_data)
Y_tr = tr_data.ACTION.to_numpy()  # Y (ACTION) = access permission (0 or 1)
X_tr = tr_data.drop("ACTION",axis=1)
Xtr = tr_data[["ROLE_DEPTNAME","ROLE_TITLE","ROLE_CODE"]]
X_tr = X_tr.to_numpy()
# Load test data
te_data = pandas.read_csv("../test.csv")

X_train, X_test, y_train, y_test = train_test_split(X_tr, Y_tr, test_size=0.2,random_state=109) # 70% training and 30% test

#Create a svm Classifier
print("[INFO] Generating the random Forrest object")

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42,max_depth = 100)
# Train the model on training data
print("[INFO] Fit the model")
rf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = rf.predict(X_test)
print(y_pred)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Loss:", metrics.log_loss(y_test,y_pred))
ids = te_data.id.to_numpy()
test = te_data.drop("id",axis=1)
test = test.to_numpy()

print("[INFO] Generating the prediction on the testing data")
y_pred_test = rf.predict(test)
print(y_pred_test)

df = pandas.DataFrame({"Id": ids, "Action": y_pred_test})
df.to_csv("prediction_rf.csv", index=False)