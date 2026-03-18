#include "GameState.h"

std::vector<int> GameState::legalMoves() const {
    std::vector<int> moves;
    moves.reserve(9);
    for (int i = 0; i < 9; i++)
        if (board[i] == Cell::Empty) moves.push_back(i);
    return moves;
}

GameState GameState::apply(int sq, Player p) const {
    GameState next = *this;
    next.set(sq, p);
    return next;
}

