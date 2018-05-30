from __future__ import division

from keras import backend as keras

from keras.preprocessing import sequence
import numpy as np
import os
import coremltools
import model
import json
import operator
import keras.callbacks
import random
import time

import signal
import time

class SignalHandler:
  stop_processing = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.stop_processing = True


# each board is a (3,3,3) matrix, where the last dimension is
# [0] == 1 for empty space
# [1] == 1 for my space
# [2] == 1 for enemy space

BOARD_SIZE = 3
MODEL_H5_NAME = "ttt.h5"
MODEL_COREML_NAME = "ttt.mlmodel"

COMPUTER_PLAYER = 0
OTHER_PLAYER = 1

GOOD_MOVE_SCORE = 1.0
TIE_MOVE_SCORE = 0.5
BAD_MOVE_SCORE = 0.0001

def PrintTrainingBoard(board, output):
	
	board_as_char = [[" "," "," "],[" "," "," "],[" "," "," "]]
	
	for x in range(0,BOARD_SIZE):
		for y in range(0,BOARD_SIZE):
			board_as_char[x][y] = " "
			if IsOn(board[x][y][COMPUTER_PLAYER]):
				board_as_char[x][y] = "O"
			elif IsOn(board[x][y][OTHER_PLAYER]):
				board_as_char[x][y] = "X"
	
	print("+---+---+---+        +-----+-----+-----+")
	print("| %s | %s | %s |        | %s | %s | %s |" % (board_as_char[0][0], board_as_char[0][1], board_as_char[0][2],   output[0][0], output[0][1], output[0][2]))
	print("+---+---+---+        +-----+-----+-----+")
	print("| %s | %s | %s |        | %s | %s | %s |" % (board_as_char[1][0], board_as_char[1][1], board_as_char[1][2],   output[1][0], output[1][1], output[1][2]))
	print("+---+---+---+        +-----+-----+-----+")
	print("| %s | %s | %s |        | %s | %s | %s |" % (board_as_char[2][0], board_as_char[2][1], board_as_char[2][2],   output[2][0], output[2][1], output[2][2]))
	print("+---+---+---+        +-----+-----+-----+")

def PrintUserBoard(board):
	
	board_as_char = [[" "," "," "],[" "," "," "],[" "," "," "]]
	
	for x in range(0,BOARD_SIZE):
		for y in range(0,BOARD_SIZE):
			board_as_char[x][y] = " "
			if IsOn(board[x][y][COMPUTER_PLAYER]):
				board_as_char[x][y] = "O"
			elif IsOn(board[x][y][OTHER_PLAYER]):
				board_as_char[x][y] = "X"
	
	print("+---+---+---+        +---+---+---+")
	print("| %s | %s | %s |        | 7 | 8 | 9 |" % (board_as_char[0][0], board_as_char[0][1], board_as_char[0][2]))
	print("+---+---+---+        +---+---+---+")
	print("| %s | %s | %s |        | 4 | 5 | 6 |" % (board_as_char[1][0], board_as_char[1][1], board_as_char[1][2]))
	print("+---+---+---+        +---+---+---+")
	print("| %s | %s | %s |        | 1 | 2 | 3 |" % (board_as_char[2][0], board_as_char[2][1], board_as_char[2][2]))
	print("+---+---+---+        +---+---+---+")

def IsOn(x):
	return x == 1

def IsEmpty(x):
	return x[0] == 0 and x[1] == 0

def Winner(a):
	for i in range(0,BOARD_SIZE):
		if np.all(a[:,i]==[1,0]):
			return 0
		if np.all(a[:,i]==[0,1]):
			return 1
		if np.all(a[i,:]==[1,0]):
			return 0
		if np.all(a[i,:]==[0,1]):
			return 1
	
	if ((a[0,0,0] and a[1,1,0] and a[2,2,0]) or
        (a[2,0,0] and a[1,1,0] and a[0,2,0])):
		return 0
	
	if ((a[0,0,1] and a[1,1,1] and a[2,2,1]) or
        (a[2,0,1] and a[1,1,1] and a[0,2,1])):
		return 1	
		
	# is the board not full?
	for x in range(0,BOARD_SIZE):
		for y in range(0,BOARD_SIZE):
			if IsEmpty(a[x][y]):
				return -1
	return 2

class TTTGameGenerator(keras.utils.Sequence):
	
	def __init__(self, batch_size=32):
		
		self.generated_input_boards = []
		self.generated_output_boards = []
		
		self.batch_size = batch_size
		self.batch_input_boards = np.zeros((self.batch_size,BOARD_SIZE,BOARD_SIZE,2), dtype=float)
		self.batch_output_boards = np.zeros((self.batch_size,BOARD_SIZE*BOARD_SIZE), dtype=float)
		
		self.generateMoreBoards(None)
			
	def __len__(self):
		return 1
		
	def __getitem__(self, _model):
		for i in range(0, self.batch_size):
			while len(self.generated_input_boards) == 0:
				self.generateMoreBoards(_model)
			
			np.copyto(self.batch_input_boards[i], self.generated_input_boards[0])
			np.copyto(self.batch_output_boards[i], self.generated_output_boards[0].flatten())
			
			del self.generated_input_boards[0]
			del self.generated_output_boards[0]
			
		return self.batch_input_boards, self.batch_output_boards
	
	
	def makeSmartMoveForPlayer(self,_model,input_board,output_board,player):
		
		if _model != None:
			local_board = np.copy(input_board)
		
			if player == 1:
				for x in range(0,BOARD_SIZE):
					for y in range(0,BOARD_SIZE):
						local_board[x][y][0] = input_board[x][y][1]
						local_board[x][y][1] = input_board[x][y][0]
		
			predictions = _model.predict(local_board.flatten().reshape((1,BOARD_SIZE,BOARD_SIZE,2)))
			ai_board = predictions.reshape((BOARD_SIZE,BOARD_SIZE))	
		
			# treat the value of the ai board as the percentage chance of choosing that spot...
			total_rand = 0
			for x in range(0,BOARD_SIZE):
				for y in range(0,BOARD_SIZE):
					if IsEmpty(local_board[x][y]) == False:
						ai_board[x][y] = 0
					total_rand += ai_board[x][y]
		
			my_rand_choice = random.random() * total_rand
		
			for x in range(0,BOARD_SIZE):
				for y in range(0,BOARD_SIZE):
					if IsEmpty(local_board[x][y]) == True:
						my_rand_choice -= ai_board[x][y]
						if my_rand_choice <= 0.0:
							input_board[x][y].fill(0)
							input_board[x][y][player] = 1
							
							output_board.fill(0)
							if player == 0:
								output_board[x][y] = GOOD_MOVE_SCORE
							return
		
		
		# if for any reason we were unable to calculate a move for ourselves,
		# make a random move
		self.makeRandomMoveForPlayer(input_board,output_board,player)
	
	def makeRandomMoveForPlayer(self,input_board,output_board,player):
		open_spaces = []
		for x in range(0,BOARD_SIZE):
			for y in range(0,BOARD_SIZE):
				if IsEmpty(input_board[x][y]):
					open_spaces.append((x,y))
		if len(open_spaces) == 0:
			return False
		
		idx = random.choice(open_spaces)
		input_board[idx[0]][idx[1]].fill(0)
		input_board[idx[0]][idx[1]][player] = 1
		
		output_board.fill(0)
		if player == 0:
			output_board[idx[0]][idx[1]] = GOOD_MOVE_SCORE
		return True
	
	def generateMoreBoards(self, _model):
		# play a game randomly. The idea here is for each turn we make a move, we
		# store the board configuration in generated_input_boards, then store
		# in our generated_output_boards in the index we are placing a "1"
		# if we end up winning the game and "0" if we end up losing the game.
				
		input_board = np.zeros((BOARD_SIZE,BOARD_SIZE,2), dtype=float)
		output_board = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=float)
				
		# we don't want to always be making the first move, so sometimes
		# we start the other player's move first.  In our setup, "0"
		# is always me and "1" is always the opponent
		player_order = [0, 1]
		if random.random() < 0.5:
			player_order = [1, 0]
		
		done = False
		while not done:
			for i in range(0,len(player_order)):
				
				if Winner(input_board) >= 0:
					done = True
					break
				
				# if its the my turn, store the state of the board before i make my choice
				if player_order[i] == 0:
					self.generated_input_boards.append(np.copy(input_board))

				self.makeSmartMoveForPlayer(_model,input_board,output_board,player_order[i])
			
				# if its the my turn, store the output of the board for my choice
				if player_order[i] == 0:
					self.generated_output_boards.append(np.copy(output_board))
		
		# we lost, all plays this time were bad (so that are 0)
		if Winner(input_board) == 1:
			for i in range(0,len(self.generated_output_boards)):
				self.generated_output_boards[i] *= BAD_MOVE_SCORE
		
		# we tied, all plays this time were only ok (so they are 0.5)
		if Winner(input_board) == 2:
			for i in range(0,len(self.generated_output_boards)):
				self.generated_output_boards[i] *= TIE_MOVE_SCORE
								

def Learn():
	
	# 1. create the model
	print("creating the model")
	_model = model.create_model(BOARD_SIZE)
	if os.path.isfile(MODEL_H5_NAME):
		_model.load_weights(MODEL_H5_NAME)

	# 2. train the model
	print("initializing the generator")
	batch_size = 1
	generator = TTTGameGenerator(batch_size)
	
	iterations = 1000000
		
	print("beginning training")
	handler = SignalHandler()
	for i in range(0,iterations):
		
		if handler.stop_processing:
			break
		
		if (i % 1000) == 0:
			print("... %s" % i)
		
		# generate a new training sample
		train,label = generator.__getitem__(_model)
		
		# our label as returned by the generator contains a board with a bunch of 0's
		# in the spaces we are not modifying, and then 0, 0.5, or 1 in the space
		# we do modify. Before we can train, we need to fill all 0 spaces with the
		# predicted outputs from our model (otherwise we are teaching the AI that
		# all plays other than the one we made are super bad, which is not the case. We
		# don't know they're bad, we just know the results of the one move we actually
		# did make)
		for j in range(0,len(train)):
			predictions = _model.predict(train[j].flatten().reshape((1,BOARD_SIZE,BOARD_SIZE,2)))
			for k in range(0,BOARD_SIZE*BOARD_SIZE):
				if label[j][k] < BAD_MOVE_SCORE:
					label[j][k] = predictions[0][k]
		
		#PrintTrainingBoard(train[0], label[0].reshape(BOARD_SIZE,BOARD_SIZE))
							
		_model.fit(train, label,
			batch_size=batch_size,
			epochs=1,
			verbose=0,
			)
	
		#again = raw_input('Continue? [y]:')
		#if again != "y":
		#	break
	
	
	_model.save(MODEL_H5_NAME)

	# 3. export to coreml
	coreml_model = coremltools.converters.keras.convert(MODEL_H5_NAME,input_names=['board'], output_names=['plays'])   
	coreml_model.author = 'Rocco Bowling'   
	coreml_model.short_description = 'Model to play tic-tac-toe'
	coreml_model.input_description['board'] = 'A normalized representation of the game board, 0 means space taken by enemy, 0.5 is empty space, 1.0 is space taken by me'
	coreml_model.save(MODEL_COREML_NAME)
	print("Conversion to coreml finished...")
	
	
#if __name__ == '__main__':
	#Learn()

def UserPlayTurn(board,space=None):
	
	space_to_coords = [ [2,0], [2,1], [2,2], 
						[1,0], [1,1], [1,2], 
						[0,0], [0,1], [0,2], ]
	
	if space == None:
		while True:
			try:
				space = int(raw_input('Choose space:'))
				if space >= 1 and space <= 9:
					space -= 1
					coords = space_to_coords[space]
					if IsEmpty(board[coords[0]][coords[1]]):
						break;
				print "Not a valid number, try again"
			except ValueError:
			    print "Not a valid number, try again"
	
	if space < 0 or space > 8:
		return False
	
	print("user space: " + str(space))
	
	coords = space_to_coords[space]
	board[coords[0]][coords[1]] = [0,1]
	return True

def AIPlayTurn(board, _model):
	
	predictions = _model.predict(board.flatten().reshape((1,BOARD_SIZE,BOARD_SIZE,2)))
	
	ai_board = predictions.reshape((BOARD_SIZE,BOARD_SIZE))
		
	PrintTrainingBoard(board, ai_board)
	
	max = 0
	for x in range(0,BOARD_SIZE):
		for y in range(0,BOARD_SIZE):
			if ai_board[x][y] > max and IsEmpty(board[x][y]):
				max = ai_board[x][y]
				coords = (x,y)

	board[coords[0]][coords[1]] = [1,0]
	

def Play():
	# simple user facing play mode to test playing against the AI
	_model = model.create_model(BOARD_SIZE)
	if os.path.isfile(MODEL_H5_NAME):
		_model.load_weights(MODEL_H5_NAME)
	
	while True:
		
		board = np.zeros((BOARD_SIZE,BOARD_SIZE,2), dtype=float)
		
		while True:
			PrintUserBoard(board)
			UserPlayTurn(board)
			if Winner(board) >= 0:
				break
			AIPlayTurn(board, _model)
			if Winner(board) >= 0:
				break
	
		print("\n\n\n")
		PrintUserBoard(board)
		winner = Winner(board)
		if winner == 1:
			print("You win!")
		if winner == 0:
			print("You lose!")
		if winner == 2:
			print("Tie game!")
		
		again = raw_input('Again? [y]:')
		if again != "y":
			break


if __name__ == '__main__':
	Play()