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

 DynamicMatrix GameState::toNNInput(const GameState& s, Player p)  {
    DynamicMatrix numericBoard(9,1);
    Cell mine = playerCell(p);
    for (int i = 0; i < 9; i++) {
        if (s.board[i] == Cell::Empty)    numericBoard.at(i,0) =  0.0f;
        else if (s.board[i] == mine)      numericBoard.at(i,0) =  1.0f;
        else                            numericBoard.at(i,0) = -1.0f;
    }
    return numericBoard;
}
