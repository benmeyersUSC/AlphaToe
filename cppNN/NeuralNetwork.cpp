//
// Created by Ben Meyers on 2/18/26.
//

#include "NeuralNetwork.h"

#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <functional>

#define MAT DynamicMatrix
#define MATVEC std::vector<MAT>


static float sigmoid(const float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static float relu(const float x) {
    return x > 0.0f ? x : 0.0f;
}
static float tanhAct(const float x) {
    return std::tanh(x);
}

// assumes z.Cols() == 1
static MAT softmax(const MAT& z) {
    if (z.Cols() != 1) {
        throw std::runtime_error("Input matrix must be column vector.");
    }
    MAT result(z.Rows(), 1);

    // extract max for numerical stability
    const float maxVal = std::accumulate(z._data().begin(), z._data().end(), -std::numeric_limits<float>::infinity(),
        [](const float a, const float b) {
        return std::max(a, b);
    });

    // summate and fill result
    float sum = 0.0f;
    std::transform(z._data().begin(), z._data().end(), result._data().begin(),
        [&sum, &maxVal](const float f) {
        const auto res = std::exp(f - maxVal);
        sum += res;
        return res;
    });

    // normalize
    std::transform(result._data().begin(), result._data().end(), result._data().begin(),
        [&sum](const float f) {
        return f/sum;
    });
    return result;
}

MAT Layer::forward(const MAT& input) const {
    const MAT z = weights * input + biases;

    return Activate(z, activation);
}

// activation derivatives
static float sigmoidPrime(const float a) {
    return a * (1.0f - a);
}
static float reluPrime(const float z) {
    return z > 0.0f ? 1.0f : 0.0f;
}
static float tanhPrime(const float a) {
    return 1.0f - a * a;
}

static MAT softmaxDelta(const MAT& a, const MAT& e) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.Rows(); i++) {
        dot += a.at(i, 0) * e.at(i, 0);
    }
    MAT d(a.Rows(), 1);
    for (size_t i = 0; i < a.Rows(); i++) {
        d.at(i, 0) = a.at(i, 0) * (e.at(i, 0) - dot);
    }
    return d;
}

void NeuralNetwork::AddLayer(const size_t inSize, const size_t outSize, const Activation act) {
    mLayers.push_back({ MAT(outSize, inSize), MAT(outSize, 1), act });
    initAdamState();
}

MAT Activate(const MAT &m, const Activation a) {
    switch (a) {
        case Activation::Softmax : return softmax(m);
        case Activation::ReLU : return m.Apply(relu);
        case Activation::Sigmoid : return m.Apply(sigmoid);
        case Activation::Tanh: return m.Apply(tanhAct);
        default: return m;
    }
}

void NeuralNetwork::initAdamState() {
    const size_t L = mLayers.size();
    mMW.assign(L, MAT(1,1));
    mVW.assign(L, MAT(1,1));
    mMB.assign(L, MAT(1,1));
    mVB.assign(L, MAT(1,1));
    for (size_t l = 0; l < L; l++) {
        const size_t r = mLayers[l].weights.Rows();
        const size_t c = mLayers[l].weights.Cols();
        const size_t numB = mLayers[l].biases.Rows();
        mMW[l] = MAT(r, c);
        mVW[l] = MAT(r, c);
        mMB[l] = MAT(numB, 1);
        mVB[l] = MAT(numB, 1);
    }
    mT = 0;
}

// forward pass and return all activations
MATVEC NeuralNetwork::ForwardAll(const MAT& input) const {
    MATVEC acts;
    acts.reserve(mLayers.size() + 1);
    acts.emplace_back(input);
    for (const auto& layer : mLayers) {
        acts.emplace_back(layer.forward(acts.back()));
    }
    return acts;
}

// forward pass
MAT NeuralNetwork::forward(const MAT& input) const {
    MAT curr = input;
    for (const auto& layer : mLayers) {
        curr = layer.forward(curr);
    }
    return curr;
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
        MAT weights(wr, wc);
        for (size_t r = 0; r < wr; r++)
            for (size_t c = 0; c < wc; c++) {
                float v;
                f.read(reinterpret_cast<char*>(&v), 4);
                weights.at(r, c) = v;
            }

        uint32_t br;
        f.read(reinterpret_cast<char*>(&br), 4);
        MAT biases(br, 1);
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

std::pair<MATVEC, MATVEC> NeuralNetwork::TrainForward(const MAT& input) const {
    MATVEC Z;
    MATVEC A;
    Z.reserve(mLayers.size());
    A.reserve(mLayers.size() + 1);
    A.emplace_back(input);
    for (const auto& [weights, biases, activation] : mLayers) {
        Z.emplace_back(weights * A.back() + biases);
        A.emplace_back(Activate(Z.back(), activation));
    }
    return {std::move(Z), std::move(A)};
}

float NeuralNetwork::CrossEntropyLoss(const MAT &result, const MAT &target) {
    std::vector<int> indices(result._data().size());
    std::iota(indices.begin(), indices.end(), 0);
    return std::accumulate(indices.begin(), indices.end(), 0.0f, [&result, &target](const float ls, const int idx) {
        const float a = std::max(result.at(idx, 0), 1e-7f);
         // expecting Target_i, seeing A_i (juiced by log)...
         // since we expect 0 everywhere but one spot, we just get log(a[correctIdx])
         return ls - target.at(idx, 0) * std::log(a);
    });
}

void NeuralNetwork::L1(MATVEC& dW, float& loss, const float l1) {
    if (l1 > 0.0f) {
        for (MAT& m : dW) {
            for (auto& w : m._data()) {
                loss += l1 * std::fabs(w);
                w += l1 * (w > 0.0f ? 1.0f : w < 0.0f ? -1.0f : 0.0f);
            }
        }
    }
}

std::tuple<MATVEC, MATVEC, MATVEC> NeuralNetwork::TrainBackward(const size_t L, const MATVEC& Z, const MATVEC& A, const MAT& target)const {
    std::vector deltas(L, MAT(1, 1));
    std::vector dW(L,     MAT(1, 1));
    std::vector dB(L,     MAT(1, 1));

    // output layer: CEL gradient is just A - Y...that means delta[targIdx] = A[targIdx] - 1, which is negative
    deltas.back() = A[L] - target; // A has L+1 elements, where [0] is input, [L] is final softmax activation

    // fundamental operation in backprop: dW_i = gradient * A_i-1
    dW.back() = deltas.back() * A[L-1].Transpose(); // [L-1] is activation being fed into final weight layer
    dB.back() = deltas.back(); // addition, full gradient flow

    // now propagate backwards
    for (int l = static_cast<int>(L - 2); l >= 0; --l){ // L - 2 is second to last weight matrix
        // this is the gradient flowing back
        // first iter:  l+1 = last WEIGHT/BIAS/GRAD, second-to-last ACTIVATION
        MAT gradient = mLayers[l+1].weights.Transpose() * deltas[l+1];

        // here we weigh/scale/gate the gradient by the activation at THIS layer
        switch (mLayers[l].activation) {
            case Activation::Sigmoid:
                // i.e. here gradient is scaled by sigmoidPrime(A-leaving-this-layer)
                deltas[l] = gradient.ElemWiseMult(A[l+1].Apply(sigmoidPrime)); break;
            case Activation::Tanh:
                deltas[l] = gradient.ElemWiseMult(A[l+1].Apply(tanhPrime));    break;
            case Activation::ReLU:
                // ReLU scales gradient depending on Z[l] (ReLU -> A[l+1] loses info from Z[l])
                deltas[l] = gradient.ElemWiseMult(Z[l].Apply(reluPrime));      break;
            case Activation::Softmax:
                // softmax in non-end case (rare), we take deriv of A-leaving-this-layer
                deltas[l] = softmaxDelta(A[l+1], gradient);                   break;
            case Activation::Input: break;
        }
        // then fundamental operation in backprop: dW_i = gradient * A_i-1
        // once again, A[l] is activation-feeding-this-layer,
        //      but dW/deltas[l] is THIS weight/gradient coming back to this layer
        dW[l] = deltas[l] * A[l].Transpose();
        dB[l] = deltas[l];
    }

    return std::make_tuple(deltas, dW, dB);
}

void NeuralNetwork::Adam(const size_t L, const float lr, const MATVEC& dW, const MATVEC& dB) {
#define PARAM_SETS std::vector<std::tuple<std::vector<float>&, const std::vector<float>&, std::vector<float>&, std::vector<float>&>> \
    {{mLayers[l].weights._data(), dW[l]._data(), mMW[l]._data(), mVW[l]._data()},\
    {mLayers[l].biases._data(),  dB[l]._data(), mMB[l]._data(), mVB[l]._data()}}

    mT++;
    const float mCorr = 1.0f / (1.0f - static_cast<float>(std::pow(mMoment1W, mT)));
    const float vCorr = 1.0f / (1.0f - static_cast<float>(std::pow(mMoment2W, mT)));

    for (size_t l = 0; l < L; l++) {
        for (auto& [params, grads, m, v] : PARAM_SETS) {
            for (size_t i = 0; i < params.size(); i++) {
                const float newM1 = m[i] * mMoment1W + mInvMoment1W * grads[i];
                const float newM2 = v[i] * mMoment2W + mInvMoment2W * grads[i] * grads[i];

                m[i] = newM1;
                v[i] = newM2;

                const float mHat = newM1 * mCorr;
                const float vHat = newM2 * vCorr;
                params[i] -= lr * mHat / (std::sqrt(vHat) + eps);
            }
        }
    }
#undef PARAM_SETS
}

TrainSnapshot NeuralNetwork::TrainStep(const MAT &input, const MAT &target, float lr, float l1) {
    const size_t L = mLayers.size();

    // FORWARD PASS
    const auto [Z, A] = TrainForward(input);

    // LOSS
    auto loss = CrossEntropyLoss(A.back(), target);

    // BACKWARD PASS
    auto [deltas, dW, dB] = TrainBackward(L, Z, A, target);

    // L1
    L1(dW, loss, l1);

    // UPDATE (Adam)
    Adam(L, lr, dW, dB);


    return {A, deltas, dW, loss};
}

void NeuralNetwork::operator<<(std::ostream& os) const {
    for (const auto& l : mLayers)
        os << l.weights.Rows() << "x" << l.weights.Cols() << "(" << ActivationName(l.activation) << ")\n";
}

#undef MAT
#undef MATVEC