#include "TerminalUI.h"
#include "game/GameRules.h"
#include "ai/Minimax.h"
#include "ai/AlphaToe.h"

#include <iostream>

// Play one game. Returns the result.
static GameResult playGame(Player humanPlayer, bool vsAlphaToe) {
    GameState state;
    Player current = Player::X;  // X always goes first
    AlphaToe alphaToe;           // constructed once (loads weights if available)

    while (true) {
        TerminalUI::printBoard(state);

        int sq;
        if (current == humanPlayer) {
            sq = TerminalUI::getHumanMove(state);
            if (sq == -1) return GameResult::Draw;  // EOF / non-interactive
        } else {
            std::cout << "AI is thinking...\n";
            if (vsAlphaToe) {
                sq = alphaToe.bestMove(state, current);
            } else {
                sq = Minimax::bestMove(state, current);
            }
        }

        state = state.apply(sq, current);

        GameResult r = GameRules::result(state);
        if (r != GameResult::Ongoing) {
            TerminalUI::printBoard(state);
            TerminalUI::printResult(r, humanPlayer);
            return r;
        }

        current = opponent(current);
    }
}

int main() {
    int mode         = TerminalUI::mainMenu();
    Player humanSide = TerminalUI::chooseSide();
    bool vsAlphaToe  = mode == 2;

    if (vsAlphaToe){
        std::cout << "\nNote: AlphaToe player not yet trained — falling back to Minimax.\n";}

    playGame(humanSide, vsAlphaToe);

    std::cout << "Thanks for playing!\n";
    return 0;
}
