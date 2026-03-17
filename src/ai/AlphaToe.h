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

    // Full MCTS result: best move + normalized visit distribution (policy training target)
    struct MoveResult {
        int move;
        std::array<float, 9> visitDist;  // sums to 1 over legal moves
    };
    MoveResult pickMove(const GameState& state, Player player, int simulations = 270) const;

    // Convenience wrapper — just the move
    int bestMove(const GameState& state, Player player, int simulations = 270) const;

    // Single training step on one sample. Returns combined loss.
    float trainStep(const DynamicMatrix& board,
                    const DynamicMatrix& policy,
                    const DynamicMatrix& value,
                    float lr,
                    float policyWeight = 1.0f,
                    float valueWeight  = 1.0f);

    void save(const std::string& path) const;
    void load(const std::string& path);
};
