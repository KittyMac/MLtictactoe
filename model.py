from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import os

BOARD_SIZE = 3
MODEL_H5_NAME = "ttt.h5"
MODEL_COREML_NAME = "ttt.mlmodel"

def doesModelExist():
	return os.path.isfile(MODEL_H5_NAME)

def createModel(loadFromDisk):

	model = Sequential()
	
	model.add(Flatten(input_shape=(BOARD_SIZE,BOARD_SIZE,2)))
	
	model.add(Dense(pow(BOARD_SIZE,6)))
	model.add(Activation('relu'))
	
	model.add(Dense(pow(BOARD_SIZE,5)))
	model.add(Activation('relu'))
	
	model.add(Dense(pow(BOARD_SIZE,4)))
	model.add(Activation('relu'))
	
	model.add(Dense(pow(BOARD_SIZE,3)))
	model.add(Activation('relu'))
	
	model.add(Dense(BOARD_SIZE*BOARD_SIZE, activation='sigmoid'))
		
	model.compile(loss='mse',
	              optimizer="adadelta")
	
	#print(model.summary())
	
	# simple user facing play mode to test playing against the AI
	if loadFromDisk and os.path.isfile(MODEL_H5_NAME):
		model.load_weights(MODEL_H5_NAME)

	return model
