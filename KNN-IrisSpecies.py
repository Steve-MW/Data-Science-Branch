#%matplotlib inline
import numpy as np
import matplotlib.pyplot
import pandas 
import mglearn 
import scipy
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

irisDataSet = load_iris()

"""
--> Essentially out data set is a dictionary, where we have the keys 
    data, target, target_names, DESCR, and feature_names.
    the key 'data' has its values as an Array which is 150 by 4 array.
    This consists of 150 data sample iris flowers i.e just iris flowers defined
    by with 4 features i.e sepel length and width, petal length and width.
    (hence 150 by 4). The key 'target' has encoddings basically, from 0-2. 0 for setosa, 
    1 for versicolor and 2 for virginica.(this is essentially the output that we are 
    looking for, in other words 'label').
    The meaning for these labels are given in target_names( setosa...). DESCR is a key whose value 
    is just a complete description of the data. Finaly feature_names is as 
    the name suggests, sepel length and width, petal length and width.
"""

xTrain,xTest,yTrain,yTest = train_test_split(irisDataSet['data'],irisDataSet['target'],random_state=0)

"""
--> We are splitting the data 75% to 25% ( training to test ).
    x having the data ( i.e the iris dimensions)
    y having the labels (i.e the output or the names of the species)
    and also the function train_test_split will do a pseudonormal 
    random split so the final 25% used for trining doesnt end up with all the same species.
    Hence our known input/output pair is split to train the Model and then the rest 
    of the known data is given as input as the output is traken to test the validity 
    or accuracy of the MODEL !
"""

# create dataframe from data in xTrain
# label the columns using the strings in irisDataSet.feature_names
irisDataFrame = pandas.DataFrame(xTrain, columns=irisDataSet.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pandas.plotting.scatter_matrix(irisDataFrame, c=yTrain, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
 # mglearn package has to imported !
 # A good understanding of the working of pandas methods such as scatter_matrix and DataFrame 
 # is required !

 # Prediction and analysis 

KNNObject = KNeighborsClassifier(n_neighbors=1)

KNNObject.fit(xTrain,yTrain)

 # Output :- KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=1, p=2,weights='uniform')

print ("Enter a set fo features( sepel length and breafth, petal length and breadth) of an iris plant ")
#xFeatures = input()

xFeatures = np.array([[4,3,2,.9]])

print("Input features = {}".format(xFeatures))

prediction = KNNObject.predict(xFeatures)

print("Predicted encoding and its subsequent value for species is :- ({},{})".format(prediction,irisDataSet['target_names'][prediction]))

yPredcited = KNNObject.predict(xTest)

print("The accuracy level of you model is : % {}".format(np.mean(yPredcited == yTest)*100))
