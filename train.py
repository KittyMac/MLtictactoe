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

BOARD_SIZE = 3
MODEL_H5_NAME = "ttt.h5"
MODEL_COREML_NAME = "ttt.mlmodel"

def Winner(a):
	for i in range(0,BOARD_SIZE):
		if np.all(a[:,i]==0):
			return 0
		if np.all(a[:,i]==1):
			return 1
		if np.all(a[i,:]==0):
			return 0
		if np.all(a[i,:]==1):
			return 1
	if np.all(a.diagonal()==0):
		return 0
	if np.all(a.diagonal()==1):
		return 1
	if np.all(np.fliplr(a).diagonal()==0):
		return 0
	if np.all(np.fliplr(a).diagonal()==1):
		return 1
	# is the board full?
	if np.any(a==0.5):
		return -1
	return 2

class TTTGameGenerator(keras.utils.Sequence):
	
	def __init__(self, batch_size=32):
		
		self.numWin = 0
		self.numLose = 0
				
		self.generated_input_boards = []
		self.generated_output_boards = []
		self.generated_boards_weights = []
		
		self.batch_size = batch_size
		self.batch_input_boards = np.zeros((self.batch_size,BOARD_SIZE*BOARD_SIZE), dtype=float)
		self.batch_output_boards = np.zeros((self.batch_size,BOARD_SIZE*BOARD_SIZE), dtype=float)
		self.batch_output_weights = np.zeros((self.batch_size,), dtype=float)
		
		self.generateMoreBoards()
			
	def __len__(self):
		return 1
		
	def __getitem__(self, index):
		for i in range(0, self.batch_size):
			while len(self.generated_input_boards) == 0:
				self.generateMoreBoards()
			
			np.copyto(self.batch_input_boards[i], self.generated_input_boards[0].flatten())
			np.copyto(self.batch_output_boards[i], self.generated_output_boards[0].flatten())
			self.batch_output_weights[i] = self.generated_boards_weights[0]
			
			del self.generated_input_boards[0]
			del self.generated_output_boards[0]
			del self.generated_boards_weights[0]
			
		return self.batch_input_boards, self.batch_output_boards, self.batch_output_weights
	
	
	def makeRandomMoveForPlayer(self,input_board,output_board,player):
		open_spaces = []
		for x in range(0,BOARD_SIZE):
			for y in range(0,BOARD_SIZE):
				if input_board[x][y] == 0.5:
					open_spaces.append((x,y))
		if len(open_spaces) == 0:
			return False
		
		idx = random.choice(open_spaces)
		input_board[idx[0]][idx[1]] = player
		output_board.fill(0)
		if player == 0:
			output_board[idx[0]][idx[1]] = 1
		return True
	
	def generateMoreBoards(self):
		# play a game randomly. The idea here is for each turn we make a move, we
		# store the board configuration in generated_input_boards, then store
		# in our generated_output_boards in the index we are placing a "1"
		# if we end up winning the game and "0" if we end up losing the game.
				
		input_board = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=float)
		output_board = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=float)
		
		# clear the initial board
		input_board.fill(0.5)
		
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
				
				self.makeRandomMoveForPlayer(input_board,output_board,player_order[i])
				
				# if its the my turn, store the output of the board for my choice
				if player_order[i] == 0:
					self.generated_output_boards.append(np.copy(output_board))
		
		# if we lost or tie, then we did not win.  Thus these are not the moves
		# we should make, 0 them out.
		if Winner(input_board) != 0:
			for i in range(0,len(self.generated_output_boards)):
				self.generated_output_boards[i].fill(0)
			self.numLose += len(self.generated_output_boards)
			weight = self.numWin / (self.numWin+self.numLose)
		else:
			self.numWin += len(self.generated_output_boards)
			weight = self.numLose / (self.numWin+self.numLose)
		
		for i in range(0,len(self.generated_output_boards)):
			self.generated_boards_weights.append(weight)
			
		#print(self.numWin, self.numLose)
			

def Learn():
	
	# 1. create the model
	print("creating the model")
	_model = model.create_model(BOARD_SIZE*BOARD_SIZE)

	# 2. train the model
	batch_size = 128
	generator = TTTGameGenerator(batch_size)
		
	for i in range(0,1000):
		# generate a new training sample
		train,label,weights = generator.__getitem__(0)
								
		_model.fit(train, label,
			batch_size=batch_size,
			epochs=1,
			verbose=1,
			sample_weight=weights
			)
	
	
	_model.save(MODEL_H5_NAME)

	# 3. export to coreml
	coreml_model = coremltools.converters.keras.convert(MODEL_H5_NAME,input_names=['board'], output_names=['plays'])   
	coreml_model.author = 'Rocco Bowling'   
	coreml_model.short_description = 'Model to play tic-tac-toe'
	coreml_model.input_description['board'] = 'A normalized representation of the game board, 0 means space taken by enemy, 0.5 is empty space, 1.0 is space taken by me'
	coreml_model.save(MODEL_COREML_NAME)
	print("Conversion to coreml finished...")
	
	

Learn()

def PrintBoard(board):
	
	board_as_char = [[" "," "," "],[" "," "," "],[" "," "," "]]
	
	for x in range(0,BOARD_SIZE):
		for y in range(0,BOARD_SIZE):
			if board[x][y] == 0:
				board_as_char[x][y] = "O"
			elif board[x][y] == 1:
				board_as_char[x][y] = "X"
			else:
				board_as_char[x][y] = " "
	
	print("+---+---+---+        +---+---+---+")
	print("| %s | %s | %s |        | 7 | 8 | 9 |" % (board_as_char[0][0], board_as_char[0][1], board_as_char[0][2]))
	print("+---+---+---+        +---+---+---+")
	print("| %s | %s | %s |        | 4 | 5 | 6 |" % (board_as_char[1][0], board_as_char[1][1], board_as_char[1][2]))
	print("+---+---+---+        +---+---+---+")
	print("| %s | %s | %s |        | 1 | 2 | 3 |" % (board_as_char[2][0], board_as_char[2][1], board_as_char[2][2]))
	print("+---+---+---+        +---+---+---+")

def UserPlayTurn(board):
	
	space_to_coords = [ [2,0], [2,1], [2,2], 
						[1,0], [1,1], [1,2], 
						[0,0], [0,1], [0,2], ]
	
	space = 0
	while True:
		try:
			space = int(raw_input('Choose space:'))
			if space >= 1 and space <= 9:
				space -= 1
				coords = space_to_coords[space]
				if board[coords[0]][coords[1]] == 0.5:
					break;
			print "Not a valid number, try again"
		except ValueError:
		    print "Not a valid number, try again"
	
	coords = space_to_coords[space]
	board[coords[0]][coords[1]] = 1

def AIPlayTurn(board, _model):
	
	predictions = _model.predict(board.flatten().reshape((1,BOARD_SIZE*BOARD_SIZE)))
	
	ai_board = predictions.reshape((BOARD_SIZE,BOARD_SIZE))
	
	print(ai_board)
	
	max = 0
	for x in range(0,BOARD_SIZE):
		for y in range(0,BOARD_SIZE):
			if ai_board[x][y] > max and board[x][y] == 0.5:
				max = ai_board[x][y]
				coords = (x,y)
	print("AI:", coords, max)
	board[coords[0]][coords[1]] = 0
	

def Play():
	# simple user facing play mode to test playing against the AI
	_model = model.create_model(BOARD_SIZE*BOARD_SIZE)
	_model.load_weights(MODEL_H5_NAME)
	
	while True:
		
		board = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=float)
		board.fill(0.5)
		
		while True:
			PrintBoard(board)
			UserPlayTurn(board)
			if Winner(board) >= 0:
				break
			AIPlayTurn(board, _model)
			if Winner(board) >= 0:
				break
	
		print("\n\n\n")
		PrintBoard(board)
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


Play()