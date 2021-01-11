from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics

train = load_files("AirSent_Train",encoding="UTF-8",shuffle=True,random_state=24)

# Feature Extraction

## Unigram Extraction
unigram_vect = CountVectorizer(binary=True)

X_train_uni = unigram_vect.fit_transform(train.data)


## Recode for step 1 neutral-polar classification
tr_step1_target = np.array([1 if i-1 else 0 for i in train.target])
step1_target_names = ["neutral","polar"]

# Model Training

## SGDClassifier with loss='log' is MaxEnt model with Gradient Descent
me1 = SGDClassifier(loss='log',penalty='elasticnet',random_state=24)
me2 = SGDClassifier(loss='log',penalty='elasticnet',random_state=24)

## Hyperparameter grid for GridSearch
### Elasticnet mixes L1 and L2 reg. 0 = L2 alone, 1 = L1 alone
params = {'l1_ratio':[0,0.125,0.25,0.5,1], 
          'alpha':[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]}

grid1 = GridSearchCV(me1,param_grid=params,cv=5).fit(X_train_uni,tr_step1_target)
grid2 = GridSearchCV(me2,param_grid=params,cv=5).fit(X_train_uni,train.target)

print(grid1.best_params_,"\n")
print(grid2.best_params_,"\n")

# Optimization on Dev Data
dev = load_files("AirSent_Dev",encoding="UTF-8",shuffle=True,random_state=24)

X_dev_uni = unigram_vect.transform(dev.data)

d_step1_target = np.array([1 if i-1 else 0 for i in dev.target])

clf1 = CalibratedClassifierCV(grid1.best_estimator_, cv='prefit').fit(X_dev_uni,d_step1_target)
clf2 = CalibratedClassifierCV(grid2.best_estimator_, cv='prefit').fit(X_dev_uni,dev.target)

test = load_files("AirSent_Test",encoding="UTF-8",shuffle=True,random_state=24)

# Step 1 Neutral-Polar Classification
X_test_uni = unigram_vect.transform(test.data)

pred1 = clf1.predict(X_test_uni)

t_step1_target = np.array([1 if i-1 else 0 for i in test.target])

print("Unigram word features only:")
print(metrics.classification_report(t_step1_target, pred1,target_names=step1_target_names))

# Collect observations classified as "polar"
step2_data = []
step2_target = []

for i in range(len(pred1)):
    if pred1[i]:
        step2_data.append(test.data[i])
        step2_target.append(test.target[i])

# Step 2 Negative-Neutral-Positive Classification
X2_test_uni = unigram_vect.transform(step2_data)

pred2 = clf2.predict(X2_test_uni)
print(metrics.classification_report(step2_target, pred2,target_names=test.target_names))
