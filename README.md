# MLtictactoe
simple NN to play TTT experiment


# Training
simple py script to generate training data from self-play samples

# Model input
a normalized array the size of the board. value of 0 means the space is mine, value of 1 means the space is theirs.  value of 0.5 means the space is no claimed.

# Model Output
a one hot array of size 9. each index should represent the probability of winning if the player chooses to place their token in said spot.  the AI always answers the questions of "where should I place my next token" (ie it only plays ones side not both).