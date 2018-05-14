from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers

def create_model(boardsize):

	model = Sequential()
	
	model.add(Dense(boardsize*4, input_shape=(boardsize,)))
	model.add(Activation('relu'))
	
	#model.add(Dense(boardsize*8))
	#model.add(Activation('relu'))
	
	#model.add(Dense(boardsize*4))
	#model.add(Activation('relu'))
	
	model.add(Dense(boardsize, activation='sigmoid'))
	
	#model.add(Dense(boardsize*2, input_shape=(boardsize,), activation='relu'))
	#model.add(Dense(boardsize*4, activation='relu'))
	#model.add(Dense(boardsize*8, activation='relu'))
	#model.add(Dense(boardsize, activation='sigmoid'))
	
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mse',
	              optimizer=sgd,
	              metrics=['accuracy'])

	return model
