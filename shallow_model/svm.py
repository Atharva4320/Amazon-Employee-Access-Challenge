#Import svm model
from sklearn import svm
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics

tr_data = pandas.read_csv("../train.csv")
Y_tr = tr_data.ACTION.to_numpy()  # Y (ACTION) = access permission (0 or 1)
X_tr = tr_data.drop("ACTION",axis=1)
X_tr = X_tr.to_numpy()
# Load test data
te_data = pandas.read_csv("../test.csv")

X_train, X_test, y_train, y_test = train_test_split(X_tr[1000:], Y_tr[1000:], test_size=0.2,random_state=109) # 70% training and 30% test

#Create a svm Classifier
print("[INFO] Generating the SVM object")
clf = svm.SVC(kernel='rbf',gamma=10) # Linear Kernel

print("[INFO] Fit the model")
#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

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
y_pred_test = clf.predict(test)
print(y_pred_test)

df = pandas.DataFrame({"Id": ids, "Action": y_pred_test})
df.to_csv("prediction.csv", index=False)