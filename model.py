from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers

def create_model(boardsize):

	model = Sequential()
	
	model.add(Flatten(input_shape=(boardsize,boardsize,2)))
	model.add(Dense(500))
	model.add(Activation('relu'))
	
	model.add(Dense(250))
	model.add(Activation('relu'))
	
	model.add(Dense(100))
	model.add(Activation('relu'))
	
	model.add(Dense(boardsize*boardsize, activation='sigmoid', use_bias=False))
		
	#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mse',
	              optimizer="rmsprop",
	              metrics=['accuracy'])
	
	
	print(model.summary())

	return model
