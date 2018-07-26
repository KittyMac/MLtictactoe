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

from model import BOARD_SIZE

PLAYER_COMPUTER = 0
PLAYER_OTHER = 1

# A GameBoard is a helper class contains the record of one turn off TTT:
# current_player is the 0,1 to represent whose turn it is to play
# board is [BOARD_SIZE,BOARD_SIZE,2] which represents the current state of the board
class GameTurn:
	def __init__(self,previousTurn):
		self.winning_player = None
		
		if previousTurn == None:
			self.current_player = PLAYER_COMPUTER if random.random() < 0.5 else PLAYER_OTHER
			self.current_player_move = None
			self.board = np.zeros((BOARD_SIZE,BOARD_SIZE,2), dtype=float)
		else:
			self.current_player = (PLAYER_COMPUTER if previousTurn.current_player == PLAYER_OTHER else PLAYER_OTHER)			
			self.current_player_move = None
			self.board = np.copy(previousTurn.board)
			self.markSpotForPlayer(previousTurn.current_player_move, previousTurn.current_player)
	
	def indexFromSpot(self,spaceXY):
		space_to_coords = [ [2,0], [2,1], [2,2], 
							[1,0], [1,1], [1,2], 
							[0,0], [0,1], [0,2] ]
		return np.where(space_to_coords==spaceXY)
	
	def coordsFromIndex(self,idx):
		space_to_coords = [ [2,0], [2,1], [2,2], 
							[1,0], [1,1], [1,2], 
							[0,0], [0,1], [0,2] ]
		return space_to_coords[idx]
	
	def spotFromIndex(self,idx):
		coords = self.coordsFromIndex(idx)
		return self.board[coords[0],coords[1]]
	
	def registerMove(self,idx):
		self.current_player_move = idx
	
	def markSpotForPlayer(self,idx,player):
		spot = self.spotFromIndex(idx)
		spot.fill(0)
		spot[player] = 1
	
	def openSpaces(self):
		open_spaces = []
		for idx in range(0,BOARD_SIZE*BOARD_SIZE):
			spot = self.spotFromIndex(idx)
			if spot[0] == 0 and spot[1] == 0:
				open_spaces.append(idx)
		return open_spaces
	
	def isValid(self):
		return self.current_player_move != None
	
	def IsOn(self,x,y,p):
		return self.board[x,y][p] == 1

	def IsEmpty(self,x,y):
		return self.board[x,y][0] == 0 and self.board[x,y][1] == 0

	def Winner(self):
		a = self.board
		
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
				if self.IsEmpty(x,y):
					return -1
		return 2
	
	def GetPredictions(self,_model):
		local_board = np.copy(self.board)
	
		if self.current_player == PLAYER_OTHER:
			for x in range(0,BOARD_SIZE):
				for y in range(0,BOARD_SIZE):
					local_board[x][y][0] = self.board[x][y][1]
					local_board[x][y][1] = self.board[x][y][0]
	
		predictions = _model.predict(local_board.flatten().reshape((1,BOARD_SIZE,BOARD_SIZE,2)))
		ai_board = predictions.reshape((BOARD_SIZE,BOARD_SIZE))
		return ai_board,local_board
	
	def GetTrainingData(self,_model):
		ai_board,local_board = self.GetPredictions(_model)
		coords = self.coordsFromIndex(self.current_player_move)
		win_value = 0.5
		if self.winning_player == self.current_player:
			win_value = 1.0
		else:
			win_value = 0.0
		#if self.current_player == PLAYER_OTHER:
		#	win_value = 1.0 - win_value
			
		ai_board[coords[0]][coords[1]] = win_value
		return local_board.reshape(1,BOARD_SIZE,BOARD_SIZE,2),ai_board.reshape(1,BOARD_SIZE*BOARD_SIZE)
		
	
	def Print(self,_model):
	
		board_as_char = [[" "," "," "],[" "," "," "],[" "," "," "]]
	
		for x in range(0,BOARD_SIZE):
			for y in range(0,BOARD_SIZE):
				board_as_char[x][y] = " "
				if self.IsOn(x,y,PLAYER_COMPUTER):
					board_as_char[x][y] = "O"
				elif self.IsOn(x,y,PLAYER_OTHER):
					board_as_char[x][y] = "X"
	
		winner_label = "draw"
		if self.winning_player == PLAYER_OTHER:
			winner_label = "X"
		if self.winning_player == PLAYER_COMPUTER:
			winner_label = "O"
		
		player_label = "O"
		if self.current_player == PLAYER_OTHER:
			player_label = "X"
		
		player_move_label = self.current_player_move
		if player_move_label != None:
			player_move_label += 1
			
		ai_board,local_board = self.GetPredictions(_model)
	
		print("")
		print("current_player", player_label, "current_player_move", player_move_label, "winning_player", winner_label)
		print("+---+---+---+        +---+---+---+        +-------+-------+-------+")
		print("| %s | %s | %s |        | 7 | 8 | 9 |        | %.3f | %.3f | %.3f |" % (board_as_char[0][0], board_as_char[0][1], board_as_char[0][2], ai_board[0][0], ai_board[0][1], ai_board[0][2], ))
		print("+---+---+---+        +---+---+---+        +-------+-------+-------+")
		print("| %s | %s | %s |        | 4 | 5 | 6 |        | %.3f | %.3f | %.3f |" % (board_as_char[1][0], board_as_char[1][1], board_as_char[1][2], ai_board[1][0], ai_board[1][1], ai_board[1][2], ))
		print("+---+---+---+        +---+---+---+        +-------+-------+-------+")
		print("| %s | %s | %s |        | 1 | 2 | 3 |        | %.3f | %.3f | %.3f |" % (board_as_char[2][0], board_as_char[2][1], board_as_char[2][2], ai_board[2][0], ai_board[2][1], ai_board[2][2], ))
		print("+---+---+---+        +---+---+---+        +-------+-------+-------+")


# the GameGenerator is responsible for providing GameTurns to the training engine.
# It does this by generating a full game's worth of GameTurns and storing them for
# ingestion into the learning process; when it runs out of turns it generates more!
class GameGenerator(keras.utils.Sequence):
	
	def __init__(self):
		self.generated_turns = []
			
	def __len__(self):
		return 1
		
	def getNextTurn(self,_model):
		
		while len(self.generated_turns) == 0:
			self.generateNewGame(_model)
		
		nextTurn = self.generated_turns[0]
		del self.generated_turns[0]
		
		return nextTurn
	
	def isEmpty(self):
		return len(self.generated_turns) == 0
	
	def makeSmartMove(self,_model,gameTurn,chooseRandom):
		
		if _model != None:
			ai_board,local_board = gameTurn.GetPredictions(_model)
		
			# treat the value of the ai board as the percentage chance of choosing that spot...
			open_spaces = gameTurn.openSpaces()
			if chooseRandom:
				total_rand = 0
				for idx in open_spaces:
					coords = gameTurn.coordsFromIndex(idx)
					total_rand += ai_board[coords[0]][coords[1]]
		
				my_rand_choice = random.random() * total_rand
				for idx in open_spaces:
					coords = gameTurn.coordsFromIndex(idx)
					my_rand_choice -= ai_board[coords[0]][coords[1]]
					if my_rand_choice <= 0.0:
						gameTurn.registerMove(idx)
						return
			else:
				best_choice = -1
				best_value = 0
				for idx in open_spaces:
					coords = gameTurn.coordsFromIndex(idx)
					space_value = ai_board[coords[0]][coords[1]]
					if space_value > best_value:
						best_value = space_value
						best_choice = idx
				gameTurn.registerMove(best_choice)
				return
		
		
		# if for any reason we were unable to calculate an informed move
		# for ourselves, then make a random move
		self.makeRandomMove(gameTurn)
	
	def makeRandomMove(self,gameTurn):
		open_spaces = gameTurn.openSpaces()
		if len(open_spaces) == 0:
			return False
		idx = random.choice(open_spaces)
		gameTurn.registerMove(idx)
		return True
	
	def generateNewGame(self, _model):
		# generate one complete game worth of turns and store each GameTurn in
		# generated_turns
		
		currentTurn = None
		
		while True:
			# get the next game turn
			currentTurn = GameTurn(currentTurn)
			# register a play
			self.makeSmartMove(_model, currentTurn, True)
			# save the turn
			if currentTurn.isValid():
				self.generated_turns.append(currentTurn)
			if currentTurn.Winner() >= 0:
				break
		
		winner = currentTurn.Winner()		
		for turn in self.generated_turns:
			turn.winning_player = winner


# Test the GameGenerator
if __name__ == '__main__':
	
	_model = model.createModel(True)
	
	generator = GameGenerator()
	
	while True:
		turn = generator.getNextTurn(_model)
		turn.Print(_model)
		if generator.isEmpty():
			break
	