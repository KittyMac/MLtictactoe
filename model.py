from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers

def create_model(boardsize):

	model = Sequential()
	
	model.add(Flatten(input_shape=(boardsize,boardsize,2)))
	model.add(Dense(pow(boardsize,5)))
	model.add(Activation('relu'))
	
	model.add(Dense(pow(boardsize,4)))
	model.add(Activation('relu'))
	
	model.add(Dense(pow(boardsize,3)))
	model.add(Activation('relu'))
	
	model.add(Dense(boardsize*boardsize, activation='sigmoid'))
		
	model.compile(loss='mse',
	              optimizer="adadelta",
	              metrics=['accuracy'])
	
	
	print(model.summary())

	return model
