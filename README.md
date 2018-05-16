# MLtictactoe
a simple NN to play tic-tac-toe experiment.


## Training
A self-training method for learning to play tic-tac-toe. Unlike some other examples I found on this topic, this features a single pass mechanism for making play decisions.


### Model input
A numpy array of (board_size,board_size,2) representing the game board. For each space, the space is empty if [0,0], captured by the AI [1,0], captured by the opponent [0,1].

### Model Output
An flattened array equivalent to (board_size,board_size), with each space representing the probability of winning if the AI chooses to play in that space; simply choosing the space with the maximum value everytime it is your turn is sufficient to play the game.


## Running

Script uses keras + various dependancies. Run the script to start and/or resume training.  Training is set to continue for 1m games, but you can ctrl-c to interupt it at any point.  After training the script enters playback mode, allowing you to play against the newly trained NN.


```
+---+---+---+        +---+---+---+
| X | X |   |        | 7 | 8 | 9 |
+---+---+---+        +---+---+---+
| X | O | X |        | 4 | 5 | 6 |
+---+---+---+        +---+---+---+
| O | O | O |        | 1 | 2 | 3 |
+---+---+---+        +---+---+---+
You lose!
```



## License

This is free software distributed under the terms of the MIT license, reproduced below. This may be used for any purpose, including commercial purposes, at absolutely no cost. No paperwork, no royalties, no GNU-like "copyleft" restrictions. Just download and enjoy.

Copyright (c) 2017 [Small Planet Digital, LLC](http://smallplanet.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
