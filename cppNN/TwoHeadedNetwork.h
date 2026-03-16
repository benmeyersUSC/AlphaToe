//
// TwoHeadedNetwork.h
// Shared-trunk neural network with separate policy and value heads,
// as used by AlphaZero-style algorithms.
//
// Architecture:
//   input → [trunk layers] → [policy head] → softmax over moves
//                          → [value head]  → tanh scalar in [-1, 1]
//

#ifndef ALPHATOE_TWOHEADEDNETWORK_H
#define ALPHATOE_TWOHEADEDNETWORK_H

#include "NeuralNetwork.h"
#include <vector>
#include <string>

struct TwoHeadedOutput {
    DynamicMatrix policy;  // [numMoves x 1], softmax probabilities
    DynamicMatrix value;   // [1 x 1], tanh score in [-1, 1]
};

class TwoHeadedNetwork {
    // Layers stored flat for efficient forward/backward.
    // trunk: indices [0, mTrunkLen)
    // policy head: indices [mTrunkLen, mTrunkLen + mPolicyLen)
    // value head: indices [mTrunkLen + mPolicyLen, ...)
    std::vector<Layer> mLayers;
    size_t mTrunkLen  = 0;
    size_t mPolicyLen = 0;
    size_t mValueLen  = 0;

    // Adam state per layer
    std::vector<DynamicMatrix> mMW, mVW, mMB, mVB;
    int mT = 0;

    void initAdamState();

    // Internal helpers: forward through a slice of mLayers.
    // Returns Z (pre-activation) and A (post-activation, A[0] = input).
    struct SliceForward {
        std::vector<DynamicMatrix> Z;
        std::vector<DynamicMatrix> A;  // A[0] = input fed to slice
    };
    SliceForward forwardSlice(size_t start, size_t end,
                              const DynamicMatrix& input) const;

    // Backward through a slice. Returns delta at the slice's input
    // (to be propagated to prior slice). Also fills dW/dB at [start..end).
    DynamicMatrix backwardSlice(size_t start, size_t end,
                                const SliceForward& fwd,
                                const DynamicMatrix& deltaIn,
                                std::vector<DynamicMatrix>& dW,
                                std::vector<DynamicMatrix>& dB);

public:
    TwoHeadedNetwork() = default;

    // Build the network from size specs.
    //   trunkSizes:  {inputSize, hidden1, hidden2, ...}  — all trunk hidden layers use ReLU
    //   policySize:  number of moves (output of policy head, uses Softmax)
    //   valueSize:   typically 1 (output of value head, uses Tanh)
    // Example for TTT: Build({9, 64, 64}, 9, 1)
    void Build(const std::vector<size_t>& trunkSizes, size_t policySize, size_t valueSize = 1);

    // Forward pass.
    [[nodiscard]] TwoHeadedOutput Forward(const DynamicMatrix& input) const;

    // One training step with Adam. Returns combined loss.
    //   policyTarget: [policySize x 1] visit-count probabilities (MCTS policy)
    //   valueTarget:  [1 x 1] game outcome in [-1, 1]
    //   policyWeight / valueWeight: loss scaling
    float TrainStep(const DynamicMatrix& input,
                    const DynamicMatrix& policyTarget,
                    const DynamicMatrix& valueTarget,
                    float lr,
                    float policyWeight = 1.0f,
                    float valueWeight  = 1.0f);

    // Binary save/load (weights + biases + architecture metadata).
    void Save(const std::string& path) const;
    void Load(const std::string& path);
};

#endif // ALPHATOE_TWOHEADEDNETWORK_H
