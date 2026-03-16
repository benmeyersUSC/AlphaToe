#pragma once
#include "GameState.h"

enum class GameResult { Ongoing, XWins, OWins, Draw };

struct GameRules {
    static GameResult result(const GameState& s);

    static bool isTerminal(const GameState& s);

    static float score(const GameState& s, Player p);
};
