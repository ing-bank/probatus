import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X, y = make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0, weights=[0.5,0.5])
n_train_samples = 1000

X_train, y_train = X[:n_train_samples], y[:n_train_samples]
X_test, y_test = X[n_train_samples:], y[n_train_samples:]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.axis('off')


clf = LogisticRegression()
clf.fit(X_train, y_train)

# For binary classification tasks predict_proba returns a matrix containing the first class proba in the first entry,
# and the second class proba in the second entry. Since there are only two classes one is just 1 - n of the other.
# The calibration_curve implementation expects just one of these classes in an array, so we index that.
y_test_predict_proba = clf.predict_proba(X_test)[:, 1]

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins=10)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-')
plt.plot([0, 1], [0, 1], '--', color='gray')

sns.despine(left=True, bottom=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.title("$LogisticRegression$ Calibration Curve", fontsize=20); pass

clf = RandomForestClassifier(max_depth = 3)
clf.fit(X_train, y_train)
y_test_predict_proba = clf.predict_proba(X_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins=10)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-')
plt.plot([0, 1], [0, 1], '--', color='gray')

sns.despine(left=True, bottom=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.title("$RF$ Sample Calibration Curve", fontsize=20); pass

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Create the example dataset and split it.
np.random.seed(42)
X, y = make_classification(n_samples=100000, n_features=20, n_informative=2, n_redundant=2, weights=[0.95,0.05])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

fig, ax = plt.subplots(1, figsize=(12, 6))

# Create an uncorrected classifier.
clf = RandomForestClassifier(max_depth=4)
clf.fit(X_train, y_train)
y_test_predict_proba = clf.predict_proba(X_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, 's-', color='red', label='Uncalibrated')

# Create a corrected classifier.
clf_sigmoid = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
clf_sigmoid.fit(X_train, y_train)
y_test_predict_proba = clf_sigmoid.predict_proba(X_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Calibrated (Platt)')
plt.plot([0, 1], [0, 1], '--', color='gray')

# Create isotomic calibration of the scores
clf_isotonic = CalibratedClassifierCV(clf, cv=3, method='isotonic')
clf_isotonic.fit(X_train, y_train)
y_test_predict_proba = clf_isotonic.predict_proba(X_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Calibrated (Isotonic)')
plt.plot([0, 1], [0, 1], '--', color='gray')

# Cubic regression from Timothee
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LogisticRegression

predIT = clf.predict_proba(X_train)[:,1]
predIT_test = clf.predict_proba(X_test)[:,1]

df = pd.DataFrame({'y': y_train, 'pred':predIT, 'pred_2': predIT**2, 'pred_3':predIT**3})
test_df = pd.DataFrame({'y': y_test, 'pred':predIT_test, 'pred_2': predIT_test**2, 'pred_3':predIT_test**3})
lr_cal = LogisticRegression()
lr_cal.fit(df[['pred','pred_2','pred_3']],df.y)

y_test_predict_proba = lr_cal.predict_proba(test_df[['pred','pred_2','pred_3']])[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Calibrated (Cube LR)')
plt.plot([0, 1], [0, 1], '--', color='gray')

sns.despine(left=True, bottom=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.gca().legend()
plt.title("$RandomForestClassifier$ Sample Calibration Curve", fontsize=20); pass
plt.show()