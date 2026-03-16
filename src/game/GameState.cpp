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

std::array<float, 9> GameState::toNNInput(Player p) const {
    std::array<float, 9> input{};
    Cell mine = playerCell(p);
    for (int i = 0; i < 9; i++) {
        if (board[i] == Cell::Empty)    input[i] =  0.0f;
        else if (board[i] == mine)      input[i] =  1.0f;
        else                            input[i] = -1.0f;
    }
    return input;
}
