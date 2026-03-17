#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include "../cppNN/DynamicMatrix.h"


// Squares are indexed 0-8, row-major:
//   0 | 1 | 2
//   ---------
//   3 | 4 | 5
//   ---------
//   6 | 7 | 8

enum class Cell : int8_t { Empty = 0, X = 1, O = -1 };
enum class Player { X, O };

inline Player opponent(Player p) { return p == Player::X ? Player::O : Player::X; }
inline Cell   playerCell(Player p) { return p == Player::X ? Cell::X : Cell::O; }

struct GameState {
    std::array<Cell, 9> board{};  // all Empty

    Cell  at(int sq) const        { return board[sq]; }
    void  set(int sq, Player p)   { board[sq] = playerCell(p); }
    bool  empty(int sq) const     { return board[sq] == Cell::Empty; }

    // all empty squares
    std::vector<int> legalMoves() const;

    // place p's mark on sq
    [[nodiscard]] GameState apply(int sq, Player p) const;

    // encodes for NN: 1.0 is me, -1.0 is opp
    static DynamicMatrix toNNInput(const GameState& s, Player p) ;

    bool operator==(const GameState& o) const { return board == o.board; }
};
