#include "TwoHeadedNetwork.h"
#include <cmath>

#define MAT DynamicMatrix

void TwoHeadedNetwork::AddLayer(size_t outSize, Activation act, Branch branch) {
    if (branch == Branch::Trunk) 
    {
        if (!mPolicy.Layers().empty() || !mValue.Layers().empty())
        {
            throw std::runtime_error("Cannot add trunk layers after head layers");
        }
        mTrunk.AddLayer(outSize, act);
    } 
    else if (branch == Branch::Policy) 
    {
        if (mPolicy.Layers().empty()) {
            size_t trunkOut = mTrunk.Layers().back().weights.Rows();
            mPolicy = NeuralNetwork(trunkOut);
        }
        mPolicy.AddLayer(outSize, act);
    } 
    else
    {
        if (mValue.Layers().empty()) {
            size_t trunkOut = mTrunk.Layers().back().weights.Rows();
            mValue = NeuralNetwork(trunkOut);
        }
        mValue.AddLayer(outSize, act);
    }
}

TwoHeadedOutput TwoHeadedNetwork::Forward(const DynamicMatrix& input) const {
    MAT trunkOut = mTrunk.forward(input);
    return { mPolicy.forward(trunkOut), mValue.forward(trunkOut) };
}

float TwoHeadedNetwork::TrainStep(const DynamicMatrix& input,
                                   const DynamicMatrix& policyTarget,
                                   const DynamicMatrix& valueTarget,
                                   float lr,
                                   float policyWeight,
                                   float valueWeight)
{
    // FORWARD (need guts)
    MAT trunkOut = mTrunk.forward(input);
    MAT policyOut = mPolicy.forward(trunkOut);
    MAT valueOut  = mValue.forward(trunkOut);

    // LOSS
    // Policy: CEL
    float policyLoss = NeuralNetwork::CrossEntropyLoss(policyOut, policyTarget);

    // Value: MSE
    float valueLoss = 0.0f;
    for (size_t i = 0; i < valueOut.Rows(); i++) {
        float diff = valueOut.at(i, 0) - valueTarget.at(i, 0);
        valueLoss += diff * diff;
    }
    valueLoss *= 0.5f; // for consistency, in MSE gradient below we omit 2 *, because we've cancelled it here

    // BACKWARD
    // Policy: softmax/CEL so just A - Y
    MAT policyDelta = policyOut - policyTarget;
    // gradient wrt input (= trunk output)
    MAT gradFromPolicy = mPolicy.TrainStepFromLastDelta(trunkOut, policyDelta * policyWeight, lr);

    // Value: tanh + MSE, so 
    //      - dL/dA = 2(A - Y) and 
    //      - dA/dZ = 1 - tanh^2(z) = 1 - A^2
    MAT valueDelta(valueOut.Rows(), 1);
    for (size_t i = 0; i < valueOut.Rows(); i++) {
        float a = valueOut.at(i, 0);
        valueDelta.at(i, 0) = (a - valueTarget.at(i, 0)) * (1.0f - a * a);  // tanh'(a) = 1 - a^2
    }
    // gradient wrt input (= trunk output)
    MAT gradFromValue = mValue.TrainStepFromLastDelta(trunkOut, valueDelta * valueWeight, lr);

    // sum of gradients of heads = gradient wrt trunk
    mTrunk.TrainStepFromGrad(input, gradFromPolicy + gradFromValue, lr);

    return policyWeight * policyLoss + valueWeight * valueLoss;
}

void TwoHeadedNetwork::Save(const std::string& path) const {
    mTrunk.Save(path + "_trunk.bin");
    mPolicy.Save(path + "_policy.bin");
    mValue.Save(path + "_value.bin");
}

void TwoHeadedNetwork::Load(const std::string& path) {
    mTrunk.Load(path + "_trunk.bin");
    mPolicy.Load(path + "_policy.bin");
    mValue.Load(path + "_value.bin");
}

#undef MAT
