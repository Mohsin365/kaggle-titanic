# Titanic Dataset(small) prediction
"""

Created on Wed Sep 11 22:28:31 2019

@author: MOHSIN AKBAR


RF-------92
accuracies.mean()
Out[4]: 0.8160512427647258

accuracies.std()
Out[5]: 0.04532380081914776

"""

# Random Forest Classification


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set   
dataset_train = pd.read_csv('train.csv')

''' dataset observation

# describe
print(dataset_train.describe)
# class distribution
print(dataset_train.groupby('class').size())


# box and whisker plots
dataset_train.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
plt.show()
# histograms
dataset_train.hist()
plt.show()


# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset_train)
plt.show()
'''



to_drop = ['PassengerId','Name','Ticket','Cabin']
dataset_train_N = dataset_train.drop(columns = to_drop,inplace = False)
#dataset_train = pd.DataFrame(dataset_train)
X_train = dataset_train_N.iloc[:, 1:8].values
# convert dataframe to object
#X_train = pd.DataFrame(X_train)

y_train = dataset_train_N.iloc[:, 0].values

# Importing the test set   
dataset_test = pd.read_csv('test.csv')
dataset_test_N = dataset_test.drop(columns = to_drop,inplace = False)

X_test = dataset_test_N.iloc[:,0:7].values
# X_test = pd.DataFrame(X_test)

###################################################################################
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer_train = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer_train = imputer_train.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer_train.transform(X_train[:, 2:3])



'''
# filling missing values ------here Embarked feature-------using KNN classification
dt = X_train
dt = pd.DataFrame(dt)

X_train_M = dt.iloc[62:828,0:6].values
y_train_M = dt.iloc[62:828,6].values

X_test_M = dt.iloc[[61,829],0:6].values
y_test_M = dt.iloc[[61,829],6].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_M = sc.fit_transform(X_train_M)
X_test_M = sc.transform(X_test_M)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_train_M = LabelEncoder()
X_train_M[:, 1] = labelencoder_train_M.fit_transform(X_train_M[:, 1]) 
labelencoder_train_Mt = LabelEncoder()
y_train_M = labelencoder_train_Mt.fit_transform(y_train_M) 

labelencoder_testT = LabelEncoder()
X_test_M[:, 1] = labelencoder_testT.fit_transform(X_test_M[:, 1]) 

from sklearn.neighbors import KNeighborsClassifier
classifier_M = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_M.fit(X_train_M, y_train_M)

# Predicting the Test set results
y_pred_M = classifier_M.predict(X_test_M)

y_train_M =pd.DataFrame(y_train_M)

# 2,0-----------S,C-------61,829
'''

X_train[61,6] = 'S'
X_train[829,6] = 'C'
# check values of missing values now
# X_train =pd.DataFrame(X_train)

# for test set missing data
imputer_test = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer_test = imputer_test.fit(X_test[:, 2:3])
imputer_test = imputer_test.fit(X_test[:, 5:6])

X_test[:, 2:3] = imputer_test.transform(X_test[:, 2:3])
X_test[:, 5:6] = imputer_test.transform(X_test[:, 5:6])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1, 6])], remainder='passthrough')
X_train = columnTransformer.fit_transform(X_train)

# avoid dummy var. trap
X_train = X_train[:,1:]

#X_test = columnTransformer.fit_transform(X_test)
columnTransformer_test = ColumnTransformer([('encoder', OneHotEncoder(), [1, 6])], remainder='passthrough')
X_test = columnTransformer_test.fit_transform(X_test)
X_test = X_test[:,1:]

###################################################################################

'''
# execute for visualization

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

'''

###################################################################################

# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 92, criterion = "gini", random_state = 7)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

my_submission = pd.DataFrame({'PassengerId': dataset_test.PassengerId, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('K-Titanic-submission.csv', index=False)
'''

# probabilities as output
y_proba = classifier.predict_proba(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [89,90,91,92,93,94,95]} ]
grid_search = GridSearchCV(refit = False,estimator = classifier,
                           param_grid = parameters,
                           # scoring = ['accuracy','explained_variance','f1'],
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_     
best_parameters = grid_search.best_params_

# cv_results = grid_search.cv_results_         # for muti-metric scoring
'''
###################################################################################


'''    UNCOMMENT PCA section first

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('TiDi(Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_pred
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('TiDi (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
'''