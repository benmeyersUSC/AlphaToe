# AlphaToe
This repository is my attempt to really claim understanding of DeepMind's AlphaZero ML approach.

Building off a previous program which uses brute force 'ai' to play optimal tic tac toe, I am going to train an AlphaZero (which I'll call AlphaToe) to hopefully play just as well!

it uses monte carlo tree search, augmented with a two-headed NN for policy and value at each state, to sample the tree.

again, with 3x3 tic tac toe, the full tree is (exceedingly) tractable but I am proving the concept of AlphaZero here.



The code is not as clean as it should be -- this was an end of summer race towards proof of concept -- but running main will work to play against AlphaToe! If you want to train it further, increase the number of iterations. 


It is a rather strong player, though there are definitely gaps in its play. I need to improve my systematic creation of diverse data for self play!