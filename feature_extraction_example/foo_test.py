from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from scipy.sparse import hstack

trcon = load_files("foo_con",encoding="UTF-8",shuffle=True,random_state=24)
trpos = load_files("foo_pos",encoding="UTF-8",shuffle=True,random_state=24)
trref = load_files("foo_ref",encoding="UTF-8",shuffle=True,random_state=24) 

con_vec = CountVectorizer(binary=True)
pos_vec = CountVectorizer()
ref_vec = CountVectorizer(binary=True)

con3_vec = CountVectorizer(binary=True,ngram_range=(1,3))


X_con = con_vec.fit_transform(trcon.data)
X_pos = pos_vec.fit_transform(trpos.data)
X_con3 = con3_vec.fit_transform(trcon.data)
X_ref = ref_vec.fit_transform(trref.data)

X3 = hstack((X_con3,X_pos,X_ref))

uni_baseline = LogisticRegression(solver="liblinear",multi_class="ovr").fit(X_con,trcon.target)
trigram = LogisticRegression(solver="liblinear",multi_class="ovr").fit(X3,trcon.target)

tcon = load_files("baz_con",encoding="UTF-8",shuffle=True,random_state=24)
tpos = load_files("baz_pos",encoding="UTF-8",shuffle=True,random_state=24)
tref = load_files("baz_ref",encoding="UTF-8",shuffle=True,random_state=24)

X_con_test = con_vec.transform(tcon.data)
X_pos_test = pos_vec.transform(tpos.data)
X_ref_test = ref_vec.transform(tref.data)

X_con3_test = con3_vec.transform(tcon.data)


X3_test = hstack((X_con3_test,X_pos_test,X_ref_test))

base_pred = uni_baseline.predict(X_con_test)
tri_pred = trigram.predict(X3_test)

print("Unigram word features only:")
print(metrics.classification_report(tcon.target, base_pred,target_names=tcon.target_names))

print("(1,3)-gram word, POS count, and reference features:")
print(metrics.classification_report(tcon.target, tri_pred,target_names=tcon.target_names))
