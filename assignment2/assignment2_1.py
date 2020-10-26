import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler
gammavalue=float(input('gamma=  '))
data = pd.read_excel('D:/LEARN/GRADUATE2020/IDA/assignment2/HW2Data.xlsx',header=None)
clf = svm.NuSVC(gamma=gammavalue)
x = data[[0,1]]
y = data[3]
#change the gamma as the parameter
x_train = x.sample(int(0.75*len(x)))
x_test = x.loc[x.index.difference(x_train.index)]
y_train = y.loc[x_train.index]
y_test = y.loc[x.index.difference(y_train.index)]

#fit model
clf.fit(x_train,y_train)
#test model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
y_test_p = clf.predict(x_test)
print('the Gamma is ',gammavalue)
print ('Accuracy of Classifier:%f'%clf.score(x_test,y_test))
print('Precision of Classifier:%f'%precision_score(y_test,y_test_p))
print('Recall of Classifier:%f'%recall_score(y_test,y_test_p))
print('Confusion matrix of Classifier:', confusion_matrix(y_test,y_test_p))


##question b
scaler =StandardScaler()
x_b = x
y_b = y
y_b_p = clf.predict(x_b)
# plot the decision function for each datapoint on the grid
xx, yy = np.meshgrid(np.linspace(0, 100, 100),
                     np.linspace(0, 100, 100))

# evaluate decision function in a grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z,  linewidths=0.5,linestyles='dashed')
plt.scatter(x_b.values[:,0], x_b.values[:,1],s=10,c=y_b_p, cmap=plt.cm.Paired,edgecolors='k')

plt.grid(True)
plt.legend(y_b_p)
plt.axis([0, 100, 0, 100])
plt.show()


# print ('Accuracy of Classifier:%f'%clf.score(x,y_b_p))
# print('Precision of Classifier:%f'%precision_score(y,y_b_p))
# print('Recall of Classifier:%f'%recall_score(y,y_b_p))
# print('Confusion matrix of Classifier:'%confusion_matrix(y,y_b_p))


# for all data in grid

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
k = list(range(0,100))
#create data for all
every1=[]
every2 = k*100
for i in k:
    every1 = [i] * 100+every1
xxinput = np.column_stack((every1,every2))
y_all = clf.predict(xxinput)
fig = plt.figure()
contours = plt.contour(xx, yy, Z,  linewidths=0.5,linestyles='dashed')
plt.scatter(xxinput[:,0],xxinput[:,1],s=10,c=y_all, cmap=plt.cm.Paired,edgecolors='k')
plt.axis([0, 100, 0, 100])
plt.show()
plt.grid(True)
plt.legend(y_all)