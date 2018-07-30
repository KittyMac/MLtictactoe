from __future__ import division

from keras import backend as keras

from keras.preprocessing import sequence
import numpy as np
import coremltools
import model
import game
import json
import operator
import keras.callbacks
import random
import time
import sys

import signal
import time

######
# allows us to used ctrl-c to end gracefully instead of just dying
######
class SignalHandler:
  stop_processing = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.stop_processing = True
######


def Learn():
	
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. train the model
	print("initializing the generator")
	batch_size = 1
	generator = game.GameGenerator()
	
	iterations = 1000000
		
	print("beginning training")
	handler = SignalHandler()
	i = 0
	while True:
		
		if handler.stop_processing:
			break
		
		print(i)
		Train(generator,_model,1000)
		i += 1000
			
		#again = raw_input('Continue? [y]:')
		#if again != "y":
		#	break
	
	
	_model.save(model.MODEL_H5_NAME)



def Train(generator,_model,n,debugPrint=False):
	
	batch_train = np.zeros((n,model.BOARD_SIZE,model.BOARD_SIZE,2), dtype=float)
	batch_label = np.zeros((n,model.BOARD_SIZE*model.BOARD_SIZE), dtype=float)
	batch_size = 32
	if n < batch_size:
		batch_size = n
	
	all_turns = []
	model_before_training = model.createModel(False)
	model_before_training.set_weights(_model.get_weights())
	
	for i in range(0,n):
		turn = generator.getNextTurn(_model)
		train,label = turn.GetTrainingData(_model)
				
		np.copyto(batch_train[i], train)
		np.copyto(batch_label[i], label)
		
		all_turns.append(turn)
	
	_model.fit(batch_train,batch_label,batch_size=batch_size,epochs=1,verbose=1)
	
	if debugPrint:
		for turn in all_turns:
			# We want to print each turn trained, the model before and after training
			turn.PrintModels(model_before_training, _model)
	
	
def UserPlayTurn(currentTurn):
	while True:
		try:
			space = int(raw_input('Choose space:'))
			if space >= 1 and space <= 9:
				space -= 1
				coords = currentTurn.coordsFromIndex(space)
				if currentTurn.IsEmpty(coords[0],coords[1]):
					break;
			print "Not a valid number, try again"
		except ValueError:
		    print "Not a valid number, try again"
	
	currentTurn.registerMove(space)	
	

def Play():
	_model = model.createModel(True)
	
	while True:
		generator = game.GameGenerator()
		
		currentTurn = None
		
		while True:
			currentTurn = game.GameTurn(currentTurn)
			
			local_winner = currentTurn.Winner()
			if local_winner >= 0:
				for turn in generator.generated_turns:
					turn.winning_player = local_winner
				break
			
			currentTurn.Print(_model)
			if currentTurn.current_player == game.PLAYER_COMPUTER:
				generator.makeSmartMove(_model,currentTurn,False)
			else:
				UserPlayTurn(currentTurn)
			
			if currentTurn.isValid():
				generator.generated_turns.append(currentTurn)
	
		print("\n\n\n")
		currentTurn.Print(_model)
		winner = currentTurn.Winner()
		if winner == 1:
			print("You win!")
		if winner == 0:
			print("You lose!")
		if winner == 2:
			print("Tie game!")
		
		Train(generator,_model,len(generator.generated_turns),True)
				
		again = raw_input('Again? [y]:')
		if again != "y":
			break
	
	_model.save(model.MODEL_H5_NAME)


if __name__ == '__main__':
	
	if sys.argv >= 2:
		if sys.argv[1] == "play":
			Play()
		if sys.argv[1] == "learn":
			Learn()
	else:
		Play()
	