#pragma once
#include "game/GameState.h"
#include <array>

struct MCTSNode {
    GameState mState;

    // nullptr means not created yet
    std::array<MCTSNode*, 9> mChildren{};

    std::array<int,   9> mVisitCounts{};
    std::array<float, 9> mValueSums{};
    int mTotalVisits = 0;

    bool mExpanded = false;  // true after network has been run on this node

    explicit MCTSNode(const GameState& state) : mState(state) {}
    ~MCTSNode();

    // mean value for sq, per this node
    float Q(int sq) const {
        // average value per visit
        return mVisitCounts[sq] > 0 ? mValueSums[sq] / mVisitCounts[sq] : 0.0f;
    }
};
