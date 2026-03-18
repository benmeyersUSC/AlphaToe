#include "AlphaToe.h"
#include "game/GameRules.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

static constexpr float C = 1.8f;

// Encode board from a player's perspective: own pieces → +1, opp → -1, empty → 0.
static Tensor<9> toTensor(const GameState& s, Player p) {
    Tensor<9> t;
    const Cell mine = playerCell(p);
    for (size_t i = 0; i < 9; ++i)
        t.flat(i) = s.board[i] == Cell::Empty ? 0.f
                  : s.board[i] == mine        ? 1.f : -1.f;
    return t;
}

std::pair<MCTSNode*,int> AlphaToe::expand(MCTSNode* node, Player player) const {
    // get policy from this node per this player
    TwoHeadedOutput out  = mNet.Forward(toTensor(node->mState, player));
    auto& policyOut = out.policy;

    // legal move indices
    auto legal = node->mState.legalMoves();

    // renormalize policy probabilities over legal moves
    float sum = 0.f;
    for (int sq : legal) sum += policyOut.flat(sq);

    int bestSq = legal[0];
    float bestProb = -1.0f;

    // for each legal move
    for (int sq : legal) {
        // add child node
        node->mChildren[sq] = new MCTSNode(node->mState.apply(sq, player));
        float prob = (sum > 0.0f) ? policyOut.flat(sq) / sum : 1.0f / static_cast<float>(legal.size());
        if (prob > bestProb)
        {
            bestProb = prob;
            bestSq = sq;
        }
    }

    node->mExpanded = true;
    return {node->mChildren[bestSq], bestSq};
}

std::pair<MCTSNode*, int> AlphaToe::ucb1(MCTSNode* node) {
    const auto legal = node->mState.legalMoves();

    int bestSq = -1;
    float bestScore = -std::numeric_limits<float>::infinity();

    for (const int sq : legal) {
        const int visits = node->mVisitCounts[sq];
        float score;
        if (visits == 0) {
            // gotta explore!
            score = std::numeric_limits<float>::infinity();
        }
        else
        {
            // average value per visit
            const float q = node->mValueSums[sq] / static_cast<float>(visits);
            // exploration constant
            const float exploration = C * std::sqrt(
                // higher if this square makes a smaller % of this state's visits
                std::log(static_cast<float>(node->mTotalVisits)) / static_cast<float>(visits)
                );
            score = q + exploration;
        }
        if (score > bestScore)
        {
            bestScore = score;
            bestSq = sq;
        }
    }

    return { node->mChildren[bestSq], bestSq };
}

void AlphaToe::backup(const std::vector<std::pair<MCTSNode*, int>>& path, const float leafValue) {
    // leafValue is value of the new leaf child from the perspective of the player TO PLAY at that node
    // but the last pair in the path is THAT NODE's parent. so to the parent, they'll reap the inverse
    // of what the child will reap if they choose to go to that node.
    float value = leafValue;
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        value = -value;   // parent wants oppo of child value!
        auto [node, sq] = path[i];
        node->mVisitCounts[sq]++;
        node->mValueSums[sq] += value;
        node->mTotalVisits++;
    }
}

AlphaToe::MoveResult AlphaToe::pickMove(const GameState& state, const Player player, const int simulations) const {
    auto* root = new MCTSNode(state);

    for (int sim = 0; sim < simulations; sim++) {
        std::vector<std::pair<MCTSNode*, int>> path;  // {node, square num taken from node}

        MCTSNode* node    = root;
        Player currPlayer = player;

        // while node has been expanded and isn't terminal, traverse and fill path
        while (node->mExpanded && !GameRules::isTerminal(node->mState)) {
            auto [child, sq] = ucb1(node);
            path.emplace_back(node, sq);
            node = child;
            currPlayer = opponent(currPlayer);
        }

        float value;
        if (GameRules::isTerminal(node->mState)) {
            value = GameRules::score(node->mState, currPlayer);
        } else {
            auto [bestChild, bestSq] = expand(node, currPlayer);
            path.emplace_back(node, bestSq);
            node = bestChild;
            currPlayer = opponent(currPlayer);

            if (GameRules::isTerminal(node->mState)) {
                value = GameRules::score(node->mState, currPlayer);
            } else {
                auto out = mNet.Forward(toTensor(node->mState, currPlayer));
                value = out.value.flat(0);
            }
        }

        backup(path, value);
    }

    // build normalized visit distribution over all 9 squares
    std::array<float, 9> visitDist{};
    float total = static_cast<float>(root->mTotalVisits);
    int bestSq  = state.legalMoves()[0];
    int bestN   = -1;
    for (int sq : state.legalMoves()) {
        visitDist[sq] = total > 0.0f ? root->mVisitCounts[sq] / total : 0.0f;
        if (root->mVisitCounts[sq] > bestN) {
            bestN  = root->mVisitCounts[sq];
            bestSq = sq;
        }
    }

    delete root;
    return { bestSq, visitDist };
}

int AlphaToe::bestMove(const GameState& state, const Player player, const int simulations) const {
    return pickMove(state, player, simulations).move;
}

float AlphaToe::trainStep(const float* board,
                          const float* policy,
                          float        value,
                          float        lr,
                          float        policyWeight,
                          float        valueWeight)
{
    Tensor<9> boardT, policyT;
    for (size_t i = 0; i < 9; ++i) { boardT.flat(i) = board[i]; policyT.flat(i) = policy[i]; }
    Tensor<1> valueT;
    valueT.flat(0) = value;

    return mNet.TrainStep(boardT, policyT, valueT, lr, policyWeight, valueWeight);
}

void AlphaToe::save(const std::string& path) const {
    fs::create_directories(fs::path(path).parent_path());
    mNet.save(path);
}
void AlphaToe::load(const std::string& path) { mNet.load(path); }
