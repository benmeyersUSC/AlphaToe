# AlphaToe
This repository is my attempt to really claim understanding of DeepMind's AlphaZero ML approach


Building off a previous program which uses brute force 'ai' to play optimal tic tac toe, I am going to train an AlphaZero (which I'll call AlphaToe) to hopefully play just as well!

it uses monte carlo tree search, augmented with a two-headed NN for policy and value at each state, to sample the tree

again, with 3x3 tic tac toe, the full tree is (exceedingly) tractable but I am proving the concept of AlphaZero here

maybe ill work towards something like a 10x10 board which would be huge and hard to win (i think the rules would need to change!)
