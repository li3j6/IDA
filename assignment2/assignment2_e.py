#Rand Index
from sklearn.cluster import KMeans
import numpy as np
from sklearn import svm
import pandas as pd

data = pd.read_excel('D:/LEARN/GRADUATE2020/IDA/assignment2/HW2Data.xlsx',header=None)
x = data[[0,1]]
y = data[[3]]
pd.to_numeric(x[0])
pd.to_numeric(x[1])
kmodel = KMeans(n_clusters=5)
yp=kmodel.fit_predict(x)
#RI = （TP+TN）/（TP+FP+FN+TN）
bimg=KMeans(n_clusters=5)
bimg.fit_predict(x.iloc[yp==0,:])
SSE1 = []
SSE1.append(bimg.inertia_)
bimg.fit_predict(x.iloc[yp==1,:])
SSE2 = []
SSE2.append(bimg.inertia_)
bimg.fit_predict(x.iloc[yp==2,:])
SSE3 = []
SSE3.append(bimg.inertia_)
bimg.fit_predict(x.iloc[yp==3,:])
SSE4 = []
SSE4.append(bimg.inertia_)
bimg.fit_predict(x.iloc[yp==4,:])
SSE5 = []
SSE5.append(bimg.inertia_)

def f_score(cluster, labels):
    TP, TN, FP, FN = 0, 0, 0, 0
    n = len(labels)
    # a lookup table
    for i in range(n):
        if i not in cluster:
            continue
        for j in range(i + 1, n):
            if j not in cluster:
                continue
            same_label = (labels[i] == labels[j])
            same_cluster = (cluster[i] == cluster[j])
            if same_cluster:
                if same_label:
                    TP += 1
                else:
                    FP += 1
            elif same_label:
                FN += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall, TP + FP + FN + TN

labels=[0,1,2,3,4]
fscore, precision, recall, tptnfpfn=f_score(yp, y)