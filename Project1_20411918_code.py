import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# Read the training data, training label and testing data from csv
data = pd.read_csv("traindata.csv", header=None)
label = pd.read_csv("trainlabel.csv", header=None)
test = pd.read_csv("testdata.csv", header=None)

# Standardise the training data and testing data by subtracting the training mean and divided by the training standard deviation
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std
test = (test - mean) / std

X_train = data.as_matrix()
Y_train = label.values.ravel()
X_test = test.as_matrix()

# Perform stratified k-fold cross validation to choose model hyper parameters
skf = StratifiedKFold(n_splits=10)
for i in np.arange(100, 600, 100):
    for j in np.arange(1, 6, 1):
        for k in np.arange(2, 6, 1):
            for l in np.arange(1, 4, 1):
                cv = 0
                tr_accuracy = 0
                tst_accuracy = 0
                for tr, tst in skf.split(X_train, Y_train):

                    # Train Test Split
                    tr_features = X_train[tr, :]
                    tr_target = Y_train[tr]
                    tst_features = X_train[tst, :]
                    tst_target = Y_train[tst]

                    # Training Random Forest Classifier
                    model = RandomForestClassifier(n_estimators=i, max_features=j, min_samples_split=k, min_samples_leaf=l)
                    model.fit(tr_features, tr_target)

                    # Measuring training and validate accuracy
                    tr_accuracy += np.mean(model.predict(tr_features) == tr_target) / 10
                    tst_accuracy += np.mean(model.predict(tst_features) == tst_target) / 10
                    cv += 1

                # Measuring validate accuracy of different parameters combination
                print "Parameters: (%d, %d, %d, %d)" %(i, j, k, l)
                print "Train Accuracy: %f, Validate Accuracy: %f" % (tr_accuracy, tst_accuracy)


# Use all the training data and the hyper parameters chosen to fit a final model
model = RandomForestClassifier(n_estimators=200, max_features=3, min_samples_split=3, min_samples_leaf=1)
model.fit(X_train, Y_train)

# Use the final model to predict the label of testing data
pd.Series(model.predict(X_test)).to_csv('Predict.csv', index=False)





