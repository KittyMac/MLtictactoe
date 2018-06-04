from __future__ import division

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty,ReferenceListProperty,ObjectProperty,VariableListProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.popup import Popup

from keras import backend as keras
import numpy as np
import os
import model
from train import TTTGameGenerator
import train
import random

class TTTSpace(Widget):
	index = VariableListProperty(2)
	def boardIndex(self):
		return self.index[0] * 3 + self.index[1]

class TTTGame(Widget):
	board_width = NumericProperty(400)
	board_height = NumericProperty(400)
	board_size = ReferenceListProperty(board_width, board_height)
	
	def __init__(self, **kwargs):
		super(TTTGame, self).__init__(**kwargs)
		
		self.model = model.create_model(train.BOARD_SIZE)
		if os.path.isfile(train.MODEL_H5_NAME):
			self.model.load_weights(train.MODEL_H5_NAME)
		
		self.generator = TTTGameGenerator(1)
		
		self.resetBoard()
	
	def on_touch_up(self,touch):
		# run through all of the spaces (children of grid) and find the one hit
		for space in self.grid.children:
			if space.label.text == "":
				if space.collide_point(touch.x,touch.y):
					if train.UserPlayTurn(self.board, space.boardIndex(), self.generator):
						self.handleAITurn()
					self.updateBoard()
	
	def handleEndOfGame(self):
		winner = train.Winner(self.board)
		if winner >= 0:
			results = "Game Over!"
			if winner == 1:
				results = "You win!"
			if winner == 0:
				results = "You lose!"
			if winner == 2:
				results = "Tie game!"
			
			popup = Popup(title='Game Over',
			    content=Label(text=results),
			    size_hint=(None, None), size=(400, 200))
			popup.open()
			
			n = self.generator.lengthOfReadyBoards()
			for i in range(0,n):
				train_set,label_set = self.generator.__getitem__(self.model)
		
				# our label as returned by the generator contains a board with a bunch of 0's
				# in the spaces we are not modifying, and then 0, 0.5, or 1 in the space
				# we do modify. Before we can train, we need to fill all 0 spaces with the
				# predicted outputs from our model (otherwise we are teaching the AI that
				# all plays other than the one we made are super bad, which is not the case. We
				# don't know they're bad, we just know the results of the one move we actually
				# did make)
				for j in range(0,len(train_set)):
					predictions = self.model.predict(train_set[j].flatten().reshape((1,train.BOARD_SIZE,train.BOARD_SIZE,2)))
					for k in range(0,train.BOARD_SIZE*train.BOARD_SIZE):
						if label_set[j][k] < train.BAD_MOVE_SCORE:
							label_set[j][k] = predictions[0][k]
		
				#PrintTrainingBoard(train_set[0], label_set[0].reshape(BOARD_SIZE,BOARD_SIZE))
							
				self.model.fit(train_set, label_set,
					batch_size=1,
					epochs=1,
					verbose=1,
					)
			
			self.resetBoard()
			
			return True
		return False
	
	def handleAITurn(self):
		if self.handleEndOfGame() == False:
			train.AIPlayTurn(self.board, self.model, self.generator)
			self.handleEndOfGame()
	
	def updateBoard(self):
		for space in self.grid.children:
			if self.board[2-space.index[0]][space.index[1]][train.COMPUTER_PLAYER] == 1:
				space.label.text = "O"
			elif self.board[2-space.index[0]][space.index[1]][train.OTHER_PLAYER] == 1:
				space.label.text = "X"
			else:
				space.label.text = ""
			
			#space.label.text = str(space.boardIndex())
		
	
	def resetBoard(self):
			self.board = np.zeros((train.BOARD_SIZE,train.BOARD_SIZE,2), dtype=float)
			
			#if random.random() < 0.5:
			#	self.handleAITurn()
			
			self.updateBoard()
	
	def on_size(self, instance, value):
		self.board_size = (self.size[0],self.size[0]) if self.size[0] < self.size[1] else (self.size[1],self.size[1])


class TTTApp(App):
	def build(self):
		game = TTTGame()
		#Clock.schedule_interval(game.update, 1.0/60.0)
		return game

if __name__ == '__main__':
	
	Config.set('graphics', 'width', '640')
	Config.set('graphics', 'height', '640')
	
	#Config.set('graphics', 'fullscreen', '0')
	#Config.write()
	
	TTTApp().run()