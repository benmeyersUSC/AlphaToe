#pragma once
#include "ai/AlphaToe.h"
#include "game/GameRules.h"
#include <vector>
#include <array>
#include <string>

// One training sample from a single move in a self-play game.
struct GameSample {
    std::array<float, 9> board;   // board encoded from that player's perspective
    std::array<float, 9> policy;  // MCTS visit distribution (sums to 1 over legal moves)
    float value;                  // game outcome from that player's perspective (+1 win, 0 draw, -1 loss)
};

class SelfPlay {
public:
    // Play numGames full self-play games. Returns all collected samples.
    static std::vector<GameSample> run(const AlphaToe& ai, int numGames, int simulations = 270);

    // Save dataset to binary file (overwrites). Creates data/ dir if needed.
    static void save(const std::vector<GameSample>& data, const std::string& path = "../data/selfplay.bin");

    // Load dataset from binary file.
    static std::vector<GameSample> load(const std::string& path = "../data/selfplay.bin");
};
