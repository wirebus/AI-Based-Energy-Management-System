# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:05:04 2019

@author: DELL
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
numpy.random.seed(7)

#load daatset
from numpy import genfromtxt
dataset = r"energy_data_01.csv"
f = open(dataset,'r')
energydata = genfromtxt(f, delimiter=',')

X = energydata[:,0:16]
Y = energydata[:,16]

#split dataset into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#create model
model = Sequential()
model.add(Dense(20, input_dim=16, activation='relu'))
model.add(Dense(20, input_dim=16, activation='relu'))
model.add(Dense(20, input_dim=16, activation='relu'))
model.add(Dense(1, input_dim=16, activation='sigmoid'))

#compile model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#fit model
model.fit(X_train, Y_train, epochs=50, batch_size=10, verbose=2, validation_data=(X_test,Y_test))

#evaluate model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(X_test)

#test model on new dataset
test1= r"new test dataset"
t = open(test1,'r')
testdata1 = genfromtxt(t, delimiter=',')
test_X = testdata1[:,0:16]
test_Y = testdata1[:,16]
predictions = model.predict(test_X)
rounded = [round(x[0])for x in predictions]
print(rounded)