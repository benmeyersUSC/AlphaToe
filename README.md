# AlphaToe
This repository is my attempt to really claim understanding of DeepMind's AlphaZero ML approach.

Building off a previous program which uses brute force 'ai' (Minimax) to play optimal tic tac toe, I am going to train an AlphaZero (which I'll call AlphaToe) to hopefully play just as well!

it uses monte carlo tree search, augmented with a two-headed NN for policy and value at each state, to sample the tree.

again, with 3x3 tic tac toe, the full tree is (exceedingly) tractable but I am proving the concept of AlphaZero here.



This codebase utilizes my Template Tensor Neural Network ([TTNN](https://github.com/benmeyersUSC/AlphaToe)) library, in which I create 
my dream: tensors and neural networks (A) whose shapes are enforced by compile-time templates and (B)
whose entire suite Inner/Outer Product and MATMUL operations flow through a generalized Einsum function!