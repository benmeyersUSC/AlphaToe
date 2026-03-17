#include "SelfPlay.h"
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <iostream>

// ── self-play ─────────────────────────────────────────────────────────────────

std::vector<GameSample> SelfPlay::run(const AlphaToe& ai, int numGames, int simulations) {
    std::vector<GameSample> dataset;
    dataset.reserve(numGames * 5);  // average ~5 moves per TTT game

    for (int g = 0; g < numGames; g++) {
        GameState state;
        Player current = Player::X;

        // pending samples — value gets filled in at game end
        struct Pending {
            std::array<float, 9> board;
            std::array<float, 9> policy;
            Player player;
        };
        std::vector<Pending> game;

        while (!GameRules::isTerminal(state)) {
            auto [move, visitDist] = ai.pickMove(state, current, simulations);

            // encode board from current player's perspective
            DynamicMatrix input = GameState::toNNInput(state, current);
            std::array<float, 9> board{};
            for (int i = 0; i < 9; i++) board[i] = input.at(i, 0);

            game.push_back({ board, visitDist, current });
            state   = state.apply(move, current);
            current = opponent(current);
        }

        // fill in outcome for every move in this game
        for (auto& s : game)
            dataset.push_back({ s.board, s.policy, GameRules::score(state, s.player) });

        if ((g + 1) % 10 == 0 || g == numGames - 1)
            std::cout << "  game " << (g + 1) << "/" << numGames
                      << "  samples so far: " << dataset.size() << "\n";
    }

    return dataset;
}

// ── binary save/load ──────────────────────────────────────────────────────────
// Format: [uint32_t numSamples] [board[9] policy[9] value] per sample

void SelfPlay::save(const std::vector<GameSample>& data, const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    uint32_t n = static_cast<uint32_t>(data.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& s : data) {
        f.write(reinterpret_cast<const char*>(s.board.data()),  sizeof(s.board));
        f.write(reinterpret_cast<const char*>(s.policy.data()), sizeof(s.policy));
        f.write(reinterpret_cast<const char*>(&s.value),        sizeof(s.value));
    }
}

std::vector<GameSample> SelfPlay::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for reading: " + path);

    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));

    std::vector<GameSample> data(n);
    for (auto& s : data) {
        f.read(reinterpret_cast<char*>(s.board.data()),  sizeof(s.board));
        f.read(reinterpret_cast<char*>(s.policy.data()), sizeof(s.policy));
        f.read(reinterpret_cast<char*>(&s.value),        sizeof(s.value));
    }
    return data;
}
