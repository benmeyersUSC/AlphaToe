//
// Created by Ben Meyers on 2/18/26.
//

#pragma once
#include <vector>
#include <string>
#include "DynamicMatrix.h"

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
    DynamicMatrix weights;  // [out_neurons x in_neurons]
    DynamicMatrix biases;   // [out_neurons x 1]
    Activation activation;

    // compute a full forward pass at this layer, taking in another matrix as input
    [[nodiscard]] DynamicMatrix forward(const DynamicMatrix& input) const;
};

struct TrainSnapshot {
    std::vector<DynamicMatrix> activations;     // [a0=input, a1, ..., aL]
    std::vector<DynamicMatrix> deltas;          // [delta1, ..., deltaL], one per layer
    std::vector<DynamicMatrix> weightGradients; // [dW1, ..., dWL], same shape as weights
    float loss = 0.0f;
};

class NeuralNetwork {
    std::vector<Layer> mLayers;

    // Adam state, moments for each weight and bias
    std::vector<DynamicMatrix> mMW, mVW;  // weight moments
    std::vector<DynamicMatrix> mMB, mVB;  // bias moments
    int mT = 0;                           // timestep

    void initAdamState();

public:
    NeuralNetwork() = default;

    // Programmatically add a layer (call in order from input to output).
    // inSize must match previous layer's outSize
    void AddLayer(size_t inSize, size_t outSize, Activation act);

    // Binary serialization — saves/loads weights and biases for all layers.
    void Save(const std::string& path) const;
    void Load(const std::string& path);

    // Forward pass. Input must be a column vector [input_size x 1].
    [[nodiscard]] DynamicMatrix forward(const DynamicMatrix& input) const;

    // Returns activations at every layer including input.
    // Result[0] = input, Result[i] = output of layer[i-1]. Size = layers + 1.
    [[nodiscard]] std::vector<DynamicMatrix> ForwardAll(const DynamicMatrix& input) const;

    [[nodiscard]] const std::vector<Layer>& Layers() const { return mLayers; }

    // One training step with Adam optimizer. Returns snapshot (activations/gradients) for visualization.
    // loss = cross-entropy. l1: optional L1 regularization.
    TrainSnapshot TrainStep(const DynamicMatrix& input, const DynamicMatrix& target, float lr, float l1 = 0.0f);

    void operator<<(std::ostream& os) const;
};

