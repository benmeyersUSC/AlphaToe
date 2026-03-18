//
// Created by Ben Meyers on 2/18/26.
//

#pragma once
#include <vector>
#include <string>
#include "DynamicMatrix.h"
#define MAT DynamicMatrix
#define MATVEC std::vector<MAT>


// available activation functions
enum class Activation {Input, Sigmoid, ReLU, Softmax, Tanh};

inline std::string_view ActivationName(const Activation& a) {
    switch (a) {
        case Activation::Input:   return "Input";
        case Activation::Sigmoid: return "Sigmoid";
        case Activation::ReLU:    return "ReLU";
        case Activation::Softmax: return "Softmax";
        case Activation::Tanh:    return "Tanh";
        default: return "Linear";
    }
}

// wrapper for weights, bias, and activation function
struct Layer {
    MAT weights;  // [out_neurons x in_neurons]
    MAT biases;   // [out_neurons x 1]
    Activation activation;

    // compute a full forward pass at this layer, taking in another matrix as input
    [[nodiscard]] MAT forward(const MAT& input) const;
};

struct TrainSnapshot {
    MATVEC activations;     // [a0=input, a1, ..., aL]
    MATVEC deltas;          // [delta1, ..., deltaL], one per layer
    MATVEC weightGradients; // [dW1, ..., dWL], same shape as weights
    float loss = 0.0f;
};

static MAT Activate(const MAT& m, Activation a);

class NeuralNetwork {
    std::vector<Layer> mLayers;
    size_t mInputSize = 0;

    // Adam state, moments for each weight and bias
    MATVEC mMW, mVW;  // weight moments
    MATVEC mMB, mVB;  // bias moments
    int mT = 0;       // timestep
    static constexpr float mMoment1W = 0.9f, mMoment2W = 0.999f, eps = 1e-8f;
    static constexpr float mInvMoment1W = 1.0f - mMoment1W, mInvMoment2W = 1.0f - mMoment2W;

    void initAdamState();
    [[nodiscard]] std::pair<MATVEC, MATVEC> TrainForward(const MAT& input) const;

    // CEL variant: computes lastDelta = A[L] - target, then calls general
    [[nodiscard]] std::tuple<MATVEC, MATVEC, MATVEC> TrainBackward(size_t L, const MATVEC& Z, const MATVEC& A, const MAT& target) const;

    // General backward: starts from a pre-computed last-layer delta 
    [[nodiscard]] std::tuple<MATVEC, MATVEC, MATVEC> TrainBackwardFromDelta(size_t L, const MATVEC& Z, const MATVEC& A, const MAT& lastDelta) const;

    static void L1(MATVEC&, float l1,  float* loss = nullptr);
    void Adam(size_t L, float lr, const MATVEC& dW, const MATVEC& dB);
    void XavierInit(MAT& weights, int inSize, int outSize);
public:
    static float CrossEntropyLoss(const MAT& result, const MAT& target);

    NeuralNetwork() = default;
    explicit NeuralNetwork(size_t inputSize) : mInputSize(inputSize) {}

    void AddLayer(size_t outSize, Activation act);

    // binary serialization of entire class
    void Save(const std::string& path) const;
    void Load(const std::string& path);

    // forward pass. input must be a column vector [input_size x 1].
    [[nodiscard]] MAT forward(const MAT& input) const;

    // computes forward pass and returns vector of activations (including input)
    [[nodiscard]] MATVEC ForwardAll(const MAT& input) const;

    [[nodiscard]] const std::vector<Layer>& Layers() const { return mLayers; }

    // train one step of CE Loss, Adam optimized, with optional L1 reg
    // returns TrainSnapshot object {vector: activations, vector: deltas, vector: weightGradients, float: loss}
    TrainSnapshot TrainStep(const MAT& input, const MAT& target, float lr, float l1 = 0.0f);

    // backward pass + Adam update from a given gradient (lastDelta)
    // returns gradient wrt input
    MAT TrainStepFromLastDelta(const MAT& input, const MAT& lastDelta, float lr, float l1 = 0.0f);

    // Train the trunk given ∂L/∂output (merged gradient from heads).
    // Converts to last-layer delta internally via activation derivative.
    void TrainStepFromGrad(const MAT& input, const MAT& grad, float lr, float l1 = 0.0f);

    void operator<<(std::ostream& os) const;
};


#undef MAT
#undef MATVEC