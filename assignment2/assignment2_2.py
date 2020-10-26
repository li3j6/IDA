import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.cm as cm
import random

data = pd.read_excel('D:/LEARN/GRADUATE2020/IDA/assignment2/HW2Data.xlsx',header=None)
x = data[[0,1]]
y = data[[3]]
kxy=data[[0,1,3]]
pd.to_numeric(x[0])
pd.to_numeric(x[1])

#change the gamma as the parameter

#fit model
def distEclud(vecA, vecB):
    'find the distance'
    return np.sum(np.power(vecA - vecB, 2))


def test_Kmeans_nclusters(data_train):
    'find the SSE value for different k'
    data_train = data_train.values
    nums = [3,5,7,9,11]
    SSE = []
    for num in nums:
        sse = 0
        kmodel = KMeans(n_clusters=num, n_jobs=4)
        kmodel.fit(data_train)
        # centers
        cluster_ceter_list = kmodel.cluster_centers_
        cluster_list = kmodel.labels_.tolist()
        for index in range(len(data)):
            cluster_num = cluster_list[index]
            sse += distEclud(data_train[index, :], cluster_ceter_list[cluster_num])
        print("number is ", num, ", the SSE is : ", sse)
        SSE.append(sse)
    return nums, SSE,kmodel
totalSSE=[]
for i in range(6):
    nums, SSE ,kmodel= test_Kmeans_nclusters(x)
    totalSSE.append(SSE)
def dispavg(sse,nums):
    avgsse=[]
    for i in range(len(sse)-1):
        k=0
        g=[]
        for j in range(len(sse)):
            k=k+sse[j][i]
            g.append(sse[j][i])
        stdg=np.std(g)
        maxv=max(g)
        minv=min(g)
        avgsse.append(k/(len(sse)))
        print('number is :',nums[i])
        print('average is:')
        print(k/(len(sse)))
        print('std is :')
        print(stdg)
        print('max is :',maxv)
        print('min is :',minv)
    return avgsse,stdg,g
print('average SSE and std for each:')
avgsse,stdg,g=dispavg(totalSSE,nums)



#b__________________________________________________
#__________________________________________________
kmodel = KMeans(n_clusters=5)
y=kmodel.fit_predict(x)
# centers
cluster_ceter_list = kmodel.cluster_centers_
cluster_list = kmodel.labels_.tolist()
#print
r1 = pd.Series(kmodel.labels_).value_counts()

r2 = pd.DataFrame(kmodel.cluster_centers_)

r = pd.concat([r2, r1], axis = 1) #
r.columns = list(x.columns) + [3] #
#plot figure
from sklearn.manifold import TSNE
tsne = TSNE()
tsne.fit_transform(x) #
tsne = pd.DataFrame(tsne.embedding_, index = x.index) #
import matplotlib.pyplot as plt
bimg=KMeans(n_clusters=5)
bimg.fit_predict(x.iloc[y==0,:])
SSE1 = []
SSE1.append(bimg.inertia_)
bimg.fit_predict(x.iloc[y==1,:])
SSE2 = []
SSE2.append(bimg.inertia_)
bimg.fit_predict(x.iloc[y==2,:])
SSE3 = []
SSE3.append(bimg.inertia_)
bimg.fit_predict(x.iloc[y==3,:])
SSE4 = []
SSE4.append(bimg.inertia_)
bimg.fit_predict(x.iloc[y==4,:])
SSE5 = []
SSE5.append(bimg.inertia_)
plt.scatter(x.iloc[y==0,0],x.iloc[y==0,1],c="orange",marker="o",label="cluster 1")
plt.scatter(x.iloc[y==1,0],x.iloc[y==1,1],c="green",marker="s",label="cluster 2")
plt.scatter(x.iloc[y==2,0],x.iloc[y==2,1],c="blue",marker="X",label="cluster 3")
plt.scatter(x.iloc[y==3,0],x.iloc[y==3,1],c="gray",marker="*",label="cluster 4")
plt.scatter(x.iloc[y==4,0],x.iloc[y==4,1],c="yellow",marker="+",label="cluster 5")
plt.scatter(bimg.cluster_centers_[0,0],bimg.cluster_centers_[0,1],s=250,marker="*",c="red" ,label="cluster 1, sse ="+str(SSE1))
plt.scatter(bimg.cluster_centers_[1,0],bimg.cluster_centers_[1,1],s=250,marker="*",c="red" ,label="cluster 2, sse ="+str(SSE2))
plt.scatter(bimg.cluster_centers_[2,0],bimg.cluster_centers_[2,1],s=250,marker="*",c="red" ,label="cluster 3, sse ="+str(SSE3))
plt.scatter(bimg.cluster_centers_[3,0],bimg.cluster_centers_[3,1],s=250,marker="*",c="red" ,label="cluster 4, sse ="+str(SSE4))
plt.scatter(bimg.cluster_centers_[4,0],bimg.cluster_centers_[4,1],s=250,marker="*",c="red" ,label="cluster 5, sse ="+str(SSE5))
plt.legend()
plt.grid()
plt.show()


