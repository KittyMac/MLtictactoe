

# winning play is set to 1, non-winning plays are 0
input_board = self.board.reshape(1,BOARD_SIZE,BOARD_SIZE,2)
output_board = np.zeros(1,(BOARD_SIZE*BOARD_SIZE), dtype=float)

coords = self.coordsFromIndex(self.current_player_move)
if self.winning_player == self.current_player:
	win_value = 1.0
else:
	win_value = 0.0
ai_board[coords[0]][coords[1]] = win_value

return input_board,output_board


# winning play is set to 1, non-winning plays are 0, tied games are set to 0.5
input_board = self.board.reshape(1,BOARD_SIZE,BOARD_SIZE,2)
output_board = np.zeros(1,(BOARD_SIZE*BOARD_SIZE), dtype=float)

coords = self.coordsFromIndex(self.current_player_move)
if self.winning_player == 2:
	win_value = 0.5
elif self.winning_player == self.current_player:
	win_value = 1.0
else:
	win_value = 0.0
ai_board[coords[0]][coords[1]] = win_value

return input_board,output_board


# winning play is set to 1, non-winning plays are 0, tied games are set to 0.5, predictions for spaces
# not affected by this turn are preserved
ai_board,local_board = self.GetPredictions(_model)
coords = self.coordsFromIndex(self.current_player_move)
if self.winning_player == 2:
	win_value = 0.5
elif self.winning_player == self.current_player:
	win_value = 1.0
else:
	win_value = 0.0
ai_board[coords[0]][coords[1]] = win_value
return local_board.reshape(1,BOARD_SIZE,BOARD_SIZE,2),ai_board.reshape(1,BOARD_SIZE*BOARD_SIZE)