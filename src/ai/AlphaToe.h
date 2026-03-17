#pragma once
#include "game/GameState.h"
#include "MCTSNode.h"
#include "TwoHeadedNetwork.h"
#include <vector>
#include <string>

class AlphaToe {
    TwoHeadedNetwork mNet;

    // Run network on node from player's perspective.
    // Adds all legal children, returns the child with the highest policy prior.
    std::pair<MCTSNode*, int> expand(MCTSNode* node, Player player) const;

    // UCB1 selection: returns (best_child, square_taken).
    static std::pair<MCTSNode*, int> ucb1(MCTSNode* node);

    // Propagate leaf value back through path.
    // path: [(node, sq_taken), ...] from root toward leaf.
    // leafValue: from the perspective of the player TO MOVE at the leaf.
    static void backup(const std::vector<std::pair<MCTSNode*, int>>& path, float leafValue);

public:
    AlphaToe();

    // Best move via MCTS + NN. Returns square 0-8.
    int bestMove(const GameState& state, Player player, int simulations = 540) const;

    void save(const std::string& path) const;
    void load(const std::string& path);
};
