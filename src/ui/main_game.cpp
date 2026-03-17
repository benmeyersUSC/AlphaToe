#include "TerminalUI.h"
#include "game/GameRules.h"
#include "ai/Minimax.h"
#include "ai/AlphaToe.h"

#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <functional>

template <typename F>
std::string timed(F&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::ostringstream os;
    if (us < 1000)
        os << us << " μs";
    else if (us < 1'000'000)
        os << us / 1000.0 << " ms";
    else
        os << us / 1'000'000.0 << " s";
    return os.str();
}

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
                auto t = timed ([&](){sq = alphaToe.bestMove(state, current);});
                std::cout << t << "\n2";
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
