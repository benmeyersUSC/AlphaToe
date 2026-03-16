//
// Created by Ben Meyers on 2/18/26.
//

#include "NeuralNetwork.h"

#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>


static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float relu(float x)    { return x > 0.0f ? x : 0.0f; }
static float tanhAct(float x) { return std::tanh(x); }

static DynamicMatrix softmax(const DynamicMatrix& z) {
    // get max
    float maxVal = z.at(0, 0);
    for (size_t i = 1; i < z.Rows(); i++) {
        maxVal = std::max(maxVal, z.at(i, 0));
    }

    DynamicMatrix out(z.Rows(), 1);
    float sum = 0.0f;
    for (size_t i = 0; i < z.Rows(); i++) {
        // subtract maxVal from power so maximum becomes e^(maxVal-maxVal) = e^0 = 1
        out.at(i, 0) = std::exp(z.at(i, 0) - maxVal);
        // summate
        sum += out.at(i, 0);
    }
    // and normalize each spot by sum
    for (size_t i = 0; i < out.Rows(); i++) {
        out.at(i, 0) /= sum;
    }
    return out;
}

DynamicMatrix Layer::forward(const DynamicMatrix& input) const {
    // get raw linear result
    const DynamicMatrix z = weights * input + biases;

    if (activation == Activation::Softmax) return softmax(z);

    const auto fn = activation == Activation::Sigmoid ? sigmoid : activation == Activation::Tanh ? tanhAct : relu;
    return z.Apply(fn);
}

// activation derivatives
static float sigmoidPrime(float a) { return a * (1.0f - a); }           // takes post-activation
static float reluPrime(float z)    { return z > 0.0f ? 1.0f : 0.0f; }  // takes pre-activation
static float tanhPrime(float a)    { return 1.0f - a * a; }             // takes post-activation

static DynamicMatrix softmaxDelta(const DynamicMatrix& a, const DynamicMatrix& e) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.Rows(); i++) {
        dot += a.at(i, 0) * e.at(i, 0);
    }
    DynamicMatrix d(a.Rows(), 1);
    for (size_t i = 0; i < a.Rows(); i++) {
        d.at(i, 0) = a.at(i, 0) * (e.at(i, 0) - dot);
    }
    return d;
}

void NeuralNetwork::AddLayer(const size_t inSize, const size_t outSize, const Activation act) {
    static std::mt19937 rng(std::random_device{}());
    mLayers.push_back({ DynamicMatrix(outSize, inSize), DynamicMatrix(outSize, 1), act });
    initAdamState();
}

void NeuralNetwork::initAdamState() {
    const size_t L = mLayers.size();
    mMW.assign(L, DynamicMatrix(1,1));
    mVW.assign(L, DynamicMatrix(1,1));
    mMB.assign(L, DynamicMatrix(1,1));
    mVB.assign(L, DynamicMatrix(1,1));
    for (size_t l = 0; l < L; l++) {
        const size_t r = mLayers[l].weights.Rows();
        const size_t c = mLayers[l].weights.Cols();
        const size_t numB = mLayers[l].biases.Rows();
        mMW[l] = DynamicMatrix(r, c);
        mVW[l] = DynamicMatrix(r, c);
        mMB[l] = DynamicMatrix(numB, 1);
        mVB[l] = DynamicMatrix(numB, 1);
    }
    mT = 0;
}

// forward pass and return all activations
std::vector<DynamicMatrix> NeuralNetwork::ForwardAll(const DynamicMatrix& input) const {
    // init activations
    std::vector<DynamicMatrix> acts;
    acts.reserve(mLayers.size() + 1);
    // input is first activation
    acts.push_back(input);
    // for each layer
    for (const auto& layer : mLayers) {
        // push back result of forward pass with previous activation as input
        acts.push_back(layer.forward(acts.back()));
    }
    return acts;
}

// forward pass
DynamicMatrix NeuralNetwork::forward(const DynamicMatrix& input) const {
    if (mLayers.empty()) {
        throw std::runtime_error("NeuralNetwork has no layers");
    }
    DynamicMatrix current = input;
    for (const auto& layer : mLayers) {
        current = layer.forward(current);
    }
    return current;
}

void NeuralNetwork::Save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

    // write num layers
    const auto numLayers = static_cast<uint32_t>(mLayers.size());
    f.write(reinterpret_cast<const char*>(&numLayers), 4);

    // for each Layer
    for (const auto&[weights, biases, activation] : mLayers) {
        // activation index
        auto act = static_cast<uint8_t>(activation);
        f.write(reinterpret_cast<const char*>(&act), 1);

        // weights
        // write weight matrix dimensions
        uint32_t wr = weights.Rows();
        uint32_t wc = weights.Cols();
        f.write(reinterpret_cast<const char*>(&wr), 4);
        f.write(reinterpret_cast<const char*>(&wc), 4);
        // write actual weights
        for (size_t r = 0; r < wr; r++)
            for (size_t c = 0; c < wc; c++) {
                float v = weights.at(r, c);
                f.write(reinterpret_cast<const char*>(&v), 4);
            }

        // biases
        uint32_t br = biases.Rows();
        f.write(reinterpret_cast<const char*>(&br), 4);
        for (size_t r = 0; r < br; r++) {
            float v = biases.at(r, 0);
            f.write(reinterpret_cast<const char*>(&v), 4);
        }
    }
}

void NeuralNetwork::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for reading: " + path);

    uint32_t numLayers;
    f.read(reinterpret_cast<char*>(&numLayers), 4);

    mLayers.clear();
    // for each layer
    for (uint32_t i = 0; i < numLayers; i++) {
        uint8_t act;
        f.read(reinterpret_cast<char*>(&act), 1);

        uint32_t wr;
        uint32_t wc;
        f.read(reinterpret_cast<char*>(&wr), 4);
        f.read(reinterpret_cast<char*>(&wc), 4);
        DynamicMatrix weights(wr, wc);
        for (size_t r = 0; r < wr; r++)
            for (size_t c = 0; c < wc; c++) {
                float v;
                f.read(reinterpret_cast<char*>(&v), 4);
                weights.at(r, c) = v;
            }

        uint32_t br;
        f.read(reinterpret_cast<char*>(&br), 4);
        DynamicMatrix biases(br, 1);
        for (size_t r = 0; r < br; r++) {
            float v;
            f.read(reinterpret_cast<char*>(&v), 4);
            biases.at(r, 0) = v;
        }

        mLayers.push_back({
            std::move(weights),
            std::move(biases),
            static_cast<Activation>(act)
        });
    }
    initAdamState();
}


TrainSnapshot NeuralNetwork::TrainStep(const DynamicMatrix& input,
                                       const DynamicMatrix& target,
                                       const float lr, const float l1)
{
    const size_t L = mLayers.size();

    // FORWARD
    // full vectors of Zs and Activations
    std::vector<DynamicMatrix> Z; Z.reserve(L);
    std::vector<DynamicMatrix> A; A.reserve(L + 1);
    A.push_back(input);
    for (const auto&[weights, biases, activation] : mLayers) {
        DynamicMatrix z = weights * A.back() + biases;
        Z.push_back(z);
        if (activation == Activation::Softmax)
            A.push_back(softmax(z));
        else if (activation == Activation::Tanh)
            A.push_back(z.Apply(tanhAct));
        else
            A.push_back(z.Apply(activation == Activation::Sigmoid ? sigmoid : relu));
    }

    // Cross Entropy Loss
    float loss = 0.0f;
    for (size_t i = 0; i < A.back().Rows(); i++) {
        const float a = std::max(A.back().at(i, 0), 1e-7f);
        // expecting Target_i, seeing A_i (juiced by log)...
        // since we expect 0 everywhere but one spot, we just get log(a[correctIdx])
        loss -= target.at(i, 0) * std::log(a);
    }

    // BACKWARD
    std::vector deltas(L, DynamicMatrix(1, 1));
    std::vector dW(L,     DynamicMatrix(1, 1));
    std::vector dB(L,     DynamicMatrix(1, 1));

    // output layer: CEL gradient is just A - Y...that means delta[targIdx] = A[targIdx] - 1, which is negative
    deltas[L-1] = A[L] - target;

    // fundamental operation in backprop: dW_i = gradient * A_i-1
    // z = W_i * A_i-1
    // gradient already takes activation = act(z) and actPrime into account
    // so gradient wrt W_i is just chain rule multiplied by A_i-1
    dW[L-1] = deltas[L-1] * A[L-1].Transpose();
    dB[L-1] = deltas[L-1];

    for (int l = static_cast<int>(L) - 2; l >= 0; --l) {
        // dL/dA_i = W_i+1 * gradient_i+1
        // because gradient_i+1 already has dW_i+1
        // and z_i+1 = A_i * W_i+1

        // so we need to bring A_i into the gradient (deltas[])
        // A_i affects through W_i+1, so we propagate gradient through that
        DynamicMatrix err = mLayers[l+1].weights.Transpose() * deltas[l+1];

        // then propagate gradient through activation derivatives
        switch (mLayers[l].activation) {
            case Activation::Sigmoid:
                deltas[l] = err.ElemWiseMult(A[l+1].Apply(sigmoidPrime)); break;
            case Activation::ReLU:
                deltas[l] = err.ElemWiseMult(Z[l].Apply(reluPrime));      break;
            case Activation::Tanh:
                deltas[l] = err.ElemWiseMult(A[l+1].Apply(tanhPrime));    break;
            case Activation::Softmax:
                deltas[l] = softmaxDelta(A[l+1], err);                   break;
            case Activation::Input: break;
        }
        // then fundamental operation in backprop: dW_i = gradient * A_i-1
        dW[l] = deltas[l] * A[l].Transpose();
        dB[l] = deltas[l];
    }

    // L1 reg
    if (l1 > 0.0f) {
        for (size_t l = 0; l < L; l++) {
            for (size_t r = 0; r < dW[l].Rows(); r++) {
                for (size_t c = 0; c < dW[l].Cols(); c++) {
                    // add to loss l1_constant * |weight|
                    const float w = mLayers[l].weights.at(r, c);
                    loss += l1 * std::fabs(w);
                    dW[l].at(r, c) += l1 * (w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f));
                }
            }
        }
    }

    // ADAM
    constexpr float fMomentW = 0.9f, sMomentW = 0.999f, eps = 1e-8f;

    mT++;
    const float correction1 = 1.0f / (1.0f - static_cast<float>(std::pow(fMomentW, mT)));
    const float correction2 = 1.0f / (1.0f - static_cast<float>(std::pow(sMomentW, mT)));

    for (size_t l = 0; l < L; l++) {
        // weights
        for (size_t r = 0; r < mLayers[l].weights.Rows(); r++) {
            for (size_t c = 0; c < mLayers[l].weights.Cols(); c++) {
                // get value
                const float val = dW[l].at(r, c);
                // get smoothed average of raw val history (captures direction)
                const float newFMoment = fMomentW * mMW[l].at(r,c) + (1.0f - fMomentW) * val;
                // get smoothed average of squared val history (captures magnitude)
                const float newSMoment = sMomentW * mVW[l].at(r,c) + (1.0f - sMomentW) * val * val;
                mMW[l].at(r,c) = newFMoment;
                mVW[l].at(r,c) = newSMoment;
                const float mHat = newFMoment * correction1;
                const float vHat = newSMoment * correction2;

                // effective step = lr * (smoothed direction) / (sqrt(steepness history) + eps)
                // weights with consistent gradients step boldly
                // noisy/steep ones tread carefully
                mLayers[l].weights.at(r,c) -= lr * mHat / (std::sqrt(vHat) + eps);
            }
        }
        // biases
        for (size_t r = 0; r < mLayers[l].biases.Rows(); r++) {
            const float val = dB[l].at(r, 0);
            const float newFMoment = fMomentW * mMB[l].at(r,0) + (1.0f - fMomentW) * val;
            const float newSMoment = sMomentW * mVB[l].at(r,0) + (1.0f - sMomentW) * val * val;
            mMB[l].at(r,0) = newFMoment;
            mVB[l].at(r,0) = newSMoment;
            const float mHat = newFMoment * correction1;
            const float vHat = newSMoment * correction2;
            mLayers[l].biases.at(r,0) -= lr * mHat / (std::sqrt(vHat) + eps);
        }
    }

    // return snapshot for viz
    return { A, deltas, dW, loss };
}

void NeuralNetwork::operator<<(std::ostream& os) const {
    for (const auto& l : mLayers)
        os << l.weights.Rows() << "x" << l.weights.Cols() << "(" << ActivationName(l.activation) << ")\n";
}
