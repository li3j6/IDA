import csv
import random
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.metrics import recall_score
import pydotplus
import os
os.environ["PATH"] += os.pathsep + 'D://Programing//python3.6//graphviz//bin'

col_names = ['pelvic_incidence',	'pelvic_tilt numeric',	'lumbar_lordosis_angle',	'sacral_slope',	'pelvic_radius',	'degree_spondylolisthesis',	'class']
# load dataset
D2 = pd.read_csv("Biomechanical_Data_2Classes.csv", header=1, names=col_names)
D2.head()

D3 = pd.read_csv("Biomechanical_Data_3Classes.csv", header=1, names=col_names)
D3.head()
def classes(data):
 df = data['class'].drop_duplicates()

 return list(df)

feature_cols = ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius','degree_spondylolisthesis']
#Feature Selection
def featureselec(data):
    feature_cols = ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
                    'degree_spondylolisthesis']
    X = data[feature_cols]  # Features
    target = ['class']
    y = data[target]  # Target variable
    return X,y
#split dataset in features and target variable
def setmodel(data,minleaf):
    X,y= featureselec(data)
       # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=80 / 309, random_state=1)
    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=minleaf)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print('Per-class precision score:', precision_score(y_test, y_pred, average=None))
    print('recall: ', recall_score(y_test, y_pred, average=None))
    return clf

# Visualizing Decision Trees
def visualtree(name,clf,data):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names= classes(data))

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(str(name)+".png")
    Image(graph.create_png())

    return
def run(data,name,minleaf):
    clf = setmodel(data,minleaf)
    visualtree(name,clf,data)
    return

run(D2,'D2_5',5)
run(D2,'D2_15',15)
run(D2,'D2_25',25)
run(D2,'D2_40',40)
run(D2,'D2_50',50)

run(D3,'D3_5',5)
run(D3,'D3_15',15)
run(D3,'D3_25',25)
run(D3,'D3_40',40)
run(D3,'D3_50',50)