# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:02:13 2020

@author: MOHSIN AKBAR
"""
'''
from keras.callbacks import Callback,ModelCheckpoint
import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    
'''
# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

'''
- Python is used by all the best Deep Learning scientists today
- it has amazing libraries (Tensorﬂow, Keras, PyTorch) developed for powerful applications...
  (Image Classiﬁcation, Computer Vision, Artiﬁcial Intelligence, ChatBots, Machine Translation, etc.).
'''

'''
Into what category does ANN fall? Supervised, Unsupervised, Reinforced Learning?
 ANNs can be all 3. Depends on how you implement the model. Examples:
 Supervised Learning: CNNs classifying images in imagenet.
 
 Unsupervised Learning: Boltzmann Machines, AutoEncoders, GANs, DC-GANS, VAE, SOMs, etc.
 
 Reinforcement: Deep Convolutional Q-Learning that plays videogames from pixel input, AlphaGO, etc.
 This branch is called "Deep Reinforcement Learning" and belongs to Artiﬁcial Intelligence".

'''

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set   
dataset_train = pd.read_csv('train.csv')

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

###################################################################################
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer_train = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer_train = imputer_train.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer_train.transform(X_train[:, 2:3])

X_train[61,6] = 'S'
X_train[829,6] = 'C'

# for test set missing data
imputer_test = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer_test = imputer_test.fit(X_test[:, 2:3])
imputer_test = imputer_test.fit(X_test[:, 5:6])

X_test[:, 2:3] = imputer_test.transform(X_test[:, 2:3])
X_test[:, 5:6] = imputer_test.transform(X_test[:, 5:6])



# Encoding categorical data
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


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# Initialising the ANN
'''
- Sequential is used to initialize the Deep Learning model as a sequence of layers...
 (as opposed to a computational graph).
 
- Dense is used to add one layer of neurons in the neural network.
'''
classifier = Sequential()

# Adding the input layer and the first hidden layer
'''
What does the rectiﬁer activation function do?
- Since Deep Learning is used to solve non linear problems, the models need to be non linear.
  And the use of the rectiﬁer function is to actually make it non linear.
  By applying it you are breaking the linearity between the output neurons and the input neurons.

What do I need to change if I have a non-binary output, e.g. win , lose and draw?
- In that case you would need to create three dummy variables for your dependent variable:
  (1,0,0) → win (0,1,0) → lose (0,0,1) → draw And therefore you need to make the following change:
     code:
         output_dim = 3 # now with the new API it is: units = 3 
         activation = ’softmax’
         loss = ’categorical_crossentropy’
'''
#classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 9))
classifier.add(Dense(units = 5, kernel_initializer = "uniform", activation = "relu", input_dim = 9)) # in new version of API

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
'''
- The cost function is a numerical estimate of how wrong our neural network predictions are.
  If we reduce the cost function, it means we are reducing the mistakes made by the neural network,
  in-turn making it predict more accurately. 

- increasing Accuracy...
  Usually ANN models take a good amount of tuning, this is usually in the form...
  of changing number of hidden layers, changing formulas, and normalizing data.
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy']) # , metrics = [tf.keras.metrics.Recall()
# tf.keras.metrics.AUC()                 tf.keras.metrics.PrecisionAtRecall(recall=0.8)         metrics=['accuracy']
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 300) # , validation_split=0.3
############################################
'''                                       
# Fitting XGBoost to the Training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
'''
##########################################
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5).astype(np.int)
y_pred = np.array(y_pred).ravel()
my_submission = pd.DataFrame({'PassengerId': dataset_test.PassengerId, 'Survived': y_pred})
# choose any filename
my_submission.to_csv('K-Titanic-submission_ANN_0.3789_0.8373.csv', index=False)

