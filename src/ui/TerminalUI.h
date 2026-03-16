#pragma once
#include "game/GameState.h"
#include "game/GameRules.h"
#include <string>


struct TerminalUI {
    static void printBoard(const GameState& s);

    static void printResult(GameResult r, Player humanPlayer);

    static int getHumanMove(const GameState& s);

    static int mainMenu();

    static Player chooseSide();
};
