#pragma once
#include "NeuralNet.hpp"

// AlphaToe-specific two-headed network.
// Architecture is intentionally hardcoded here — this is a creature of the
// AlphaToe implementation, not of the generic cppNN library.
//
//   Trunk:  9 → 54 (ReLU) → 27 (ReLU)     shared board encoder
//   Policy: 27 → 27 (ReLU) → 9 (Softmax)  move probabilities over 9 squares
//   Value:  27 → 27 (ReLU) → 1 (Tanh)     scalar outcome in [-1, 1]

struct TwoHeadedOutput {
    Tensor<9> policy;   // softmax distribution over 9 squares
    Tensor<1> value;    // tanh score in [-1, 1]
};

class TwoHeadedNetwork {
    using Trunk  = NeuralNetwork<9,  54, 27>;
    using Policy = NeuralNetwork<27, 27, 9 >;
    using Value  = NeuralNetwork<27, 27, 1 >;

    Trunk  mTrunk  { ActivationFunction::ReLU, ActivationFunction::ReLU    };
    Policy mPolicy { ActivationFunction::ReLU, ActivationFunction::Softmax };
    Value  mValue  { ActivationFunction::ReLU, ActivationFunction::Tanh    };

public:
    TwoHeadedNetwork() = default;

    [[nodiscard]] TwoHeadedOutput Forward(const Tensor<9>& x) const {
        const auto trunkOut = mTrunk.Forward(x);
        return { mPolicy.Forward(trunkOut), mValue.Forward(trunkOut) };
    }

    // Single training step. Returns weighted combined loss.
    //
    // policyTarget: MCTS visit distribution (sums to 1 over legal moves)
    // valueTarget:  game outcome from this player's perspective (+1 win, -1 loss, 0 draw)
    float TrainStep(const Tensor<9>& x,
                    const Tensor<9>& policyTarget,
                    const Tensor<1>& valueTarget,
                    float lr,
                    float policyWeight = 1.f,
                    float valueWeight  = 1.f)
    {
        // --- Forward ---
        const auto trunkOut  = mTrunk.Forward(x);
        const auto policyOut = mPolicy.Forward(trunkOut);
        const auto valueOut  = mValue.Forward(trunkOut);

        // --- Policy: softmax + CEL.  Loss = -Σ y_i log(a_i).  Delta = a - y. ---
        const float policyLoss  = CrossEntropyLoss(policyOut, policyTarget);
        const auto  policyDelta = policyOut - policyTarget;

        // --- Value: tanh + MSE.  Loss = ½(a-y)².  Delta = (a-y)*(1-a²). ---
        const float a = valueOut.flat(0), y = valueTarget.flat(0);
        const float valueLoss = 0.5f * (a - y) * (a - y);
        Tensor<1> valueDelta;
        valueDelta.flat(0) = (a - y) * (1.f - a * a);   // tanh'(a) = 1 - a²

        // --- Backward through each head → gradient wrt trunk output ---
        const auto gradFromPolicy = mPolicy.TrainStepLogits(trunkOut, policyDelta * policyWeight, lr);
        const auto gradFromValue  = mValue.TrainStepLogits(trunkOut,  valueDelta  * valueWeight,  lr);

        // --- Backward through trunk from merged head gradients ---
        mTrunk.TrainStep(x, gradFromPolicy + gradFromValue, lr);

        return policyWeight * policyLoss + valueWeight * valueLoss;
    }

    // Save / load — available when models are ready to persist.
    void save(const std::string& path) const {
        mTrunk.Save(path  + "_trunk.bin");
        mPolicy.Save(path + "_policy.bin");
        mValue.Save(path  + "_value.bin");
    }
    void load(const std::string& path) {
        mTrunk.Load(path  + "_trunk.bin");
        mPolicy.Load(path + "_policy.bin");
        mValue.Load(path  + "_value.bin");
    }
};
