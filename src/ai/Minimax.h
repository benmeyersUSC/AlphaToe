#pragma once
#include "game/GameState.h"


struct Minimax {
    static int   bestMove(const GameState& s, Player player);
    private:
    static float search(const GameState& s, Player rootPlayer, bool maximizing);
};
