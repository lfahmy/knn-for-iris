# Load libraries
from pandas import *
from math import sqrt
from knn import *
# from pandas.tools.plotting import scatter_matrix
# import matplotlib.pyplot as plt
# from sklearn import model_selection
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

random = dataset.sample(n=150)
labeled = random[0:100]
unlabeled = random[100:]
solution = unlabeled
unlabeled = unlabeled.drop('class', axis=1)
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
firstTry = Knn(labeled, unlabeled, classes)
response = firstTry.classifyAll()
firstTry.accuracy(response, solution)