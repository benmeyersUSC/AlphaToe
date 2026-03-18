#pragma once
#include "NeuralNetwork.h"
#include <cassert>

enum class Branch { Trunk, Policy, Value };

struct TwoHeadedOutput {
    DynamicMatrix policy;  // [numMoves x 1], softmax probabilities
    DynamicMatrix value;   // [1 x 1], tanh score for value [-1, 1]
};

// Shared-trunk network with two output heads (policy + value), as used by AlphaToe
//
// Build by calling AddLayer in order: trunk first, then heads
// Once any head layer is added, trunk is locked
class TwoHeadedNetwork {
    NeuralNetwork mTrunk, mPolicy, mValue;

public:
    TwoHeadedNetwork() = default;
    explicit TwoHeadedNetwork(size_t inputSize) : mTrunk(inputSize) {}

    // Add a layer to the specified branch
    // - Trunk must be fully built before any head layers are added
    // - Input size (for new trunk layer or for new head layer) is inferred automatically from prev output
    void AddLayer(size_t outSize, Activation act, Branch branch);

    [[nodiscard]] TwoHeadedOutput Forward(const DynamicMatrix& input) const;

    // Single backward + Adam update
    // Returns combined loss
    // Targets:
    //   policyTarget: [numMoves x 1]
    //   valueTarget:  [1 x 1] game outcome [-1, 1]
    float TrainStep(const DynamicMatrix& input,
                    const DynamicMatrix& policyTarget,
                    const DynamicMatrix& valueTarget,
                    float lr,
                    float policyWeight = 1.0f,
                    float valueWeight  = 1.0f);

    // three files: path_trunk.bin, path_policy.bin, path_value.bin
    void Save(const std::string& path) const;
    void Load(const std::string& path);
};
