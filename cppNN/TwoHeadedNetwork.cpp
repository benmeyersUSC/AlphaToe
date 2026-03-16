//
// TwoHeadedNetwork.cpp
//

#include "TwoHeadedNetwork.h"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>


// ---- activation helpers (mirrored from NeuralNetwork.cpp) ----
static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float relu(float x)    { return x > 0.0f ? x : 0.0f; }
static float tanhAct(float x) { return std::tanh(x); }

static DynamicMatrix softmax(const DynamicMatrix& z) {
    float maxVal = z.at(0, 0);
    for (size_t i = 1; i < z.Rows(); i++) maxVal = std::max(maxVal, z.at(i, 0));
    DynamicMatrix out(z.Rows(), 1);
    float sum = 0.0f;
    for (size_t i = 0; i < z.Rows(); i++) { out.at(i,0) = std::exp(z.at(i,0) - maxVal); sum += out.at(i,0); }
    for (size_t i = 0; i < out.Rows(); i++) out.at(i,0) /= sum;
    return out;
}

static float sigmoidPrime(float a) { return a * (1.0f - a); }
static float reluPrime(float z)    { return z > 0.0f ? 1.0f : 0.0f; }
static float tanhPrime(float a)    { return 1.0f - a * a; }

static DynamicMatrix softmaxDelta(const DynamicMatrix& a, const DynamicMatrix& e) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.Rows(); i++) dot += a.at(i,0) * e.at(i,0);
    DynamicMatrix d(a.Rows(), 1);
    for (size_t i = 0; i < a.Rows(); i++) d.at(i,0) = a.at(i,0) * (e.at(i,0) - dot);
    return d;
}

// Apply activation for a layer
static DynamicMatrix applyAct(const DynamicMatrix& z, Activation act) {
    if (act == Activation::Softmax) return softmax(z);
    if (act == Activation::Tanh)    return z.Apply(tanhAct);
    if (act == Activation::Sigmoid) return z.Apply(sigmoid);
    return z.Apply(relu); // ReLU default
}


// ---- Xavier init ----
static DynamicMatrix xavierWeights(size_t out, size_t in, std::mt19937& rng) {
    float limit = std::sqrt(6.0f / static_cast<float>(in + out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    DynamicMatrix w(out, in);
    for (size_t r = 0; r < out; r++)
        for (size_t c = 0; c < in; c++)
            w.at(r, c) = dist(rng);
    return w;
}

static void addLayer(std::vector<Layer>& layers, size_t in, size_t out,
                     Activation act, std::mt19937& rng) {
    layers.push_back({ xavierWeights(out, in, rng), DynamicMatrix(out, 1), act });
}


// ---- Build ----
void TwoHeadedNetwork::Build(const std::vector<size_t>& trunkSizes,
                              size_t policySize, size_t valueSize)
{
    if (trunkSizes.size() < 2)
        throw std::runtime_error("trunkSizes needs at least {inputSize, hiddenSize}");

    std::mt19937 rng(std::random_device{}());
    mLayers.clear();

    // trunk: all hidden layers use ReLU
    for (size_t i = 1; i < trunkSizes.size(); i++)
        addLayer(mLayers, trunkSizes[i-1], trunkSizes[i], Activation::ReLU, rng);
    mTrunkLen = trunkSizes.size() - 1;

    size_t trunkOut = trunkSizes.back();

    // policy head: single dense layer → softmax
    addLayer(mLayers, trunkOut, policySize, Activation::Softmax, rng);
    mPolicyLen = 1;

    // value head: single dense layer → tanh
    addLayer(mLayers, trunkOut, valueSize, Activation::Tanh, rng);
    mValueLen = 1;

    initAdamState();
}


// ---- Adam state ----
void TwoHeadedNetwork::initAdamState() {
    const size_t L = mLayers.size();
    mMW.assign(L, DynamicMatrix(1,1));
    mVW.assign(L, DynamicMatrix(1,1));
    mMB.assign(L, DynamicMatrix(1,1));
    mVB.assign(L, DynamicMatrix(1,1));
    for (size_t l = 0; l < L; l++) {
        size_t r = mLayers[l].weights.Rows(), c = mLayers[l].weights.Cols();
        size_t rb = mLayers[l].biases.Rows();
        mMW[l] = DynamicMatrix(r, c);
        mVW[l] = DynamicMatrix(r, c);
        mMB[l] = DynamicMatrix(rb, 1);
        mVB[l] = DynamicMatrix(rb, 1);
    }
    mT = 0;
}


// ---- forward slice ----
TwoHeadedNetwork::SliceForward TwoHeadedNetwork::forwardSlice(
    size_t start, size_t end, const DynamicMatrix& input) const
{
    SliceForward sf;
    sf.A.push_back(input);
    for (size_t l = start; l < end; l++) {
        DynamicMatrix z = mLayers[l].weights * sf.A.back() + mLayers[l].biases;
        sf.Z.push_back(z);
        sf.A.push_back(applyAct(z, mLayers[l].activation));
    }
    return sf;
}


// ---- forward ----
TwoHeadedOutput TwoHeadedNetwork::Forward(const DynamicMatrix& input) const {
    // trunk
    auto trunk = forwardSlice(0, mTrunkLen, input);
    const DynamicMatrix& trunkOut = trunk.A.back();

    // heads
    auto policyFwd = forwardSlice(mTrunkLen, mTrunkLen + mPolicyLen, trunkOut);
    auto valueFwd  = forwardSlice(mTrunkLen + mPolicyLen, mTrunkLen + mPolicyLen + mValueLen, trunkOut);

    return { policyFwd.A.back(), valueFwd.A.back() };
}


// ---- backward slice ----
// Returns the delta to propagate into the previous slice (i.e., w.r.t. slice input).
DynamicMatrix TwoHeadedNetwork::backwardSlice(
    size_t start, size_t end,
    const SliceForward& fwd,
    const DynamicMatrix& deltaIn,        // delta at the slice OUTPUT
    std::vector<DynamicMatrix>& dW,
    std::vector<DynamicMatrix>& dB)
{
    size_t len = end - start;
    std::vector<DynamicMatrix> deltas(len, DynamicMatrix(1,1));

    // output layer of this slice: deltaIn is already the correct delta
    // (caller responsible for computing output delta, e.g., a-y for cross-entropy+softmax)
    deltas[len-1] = deltaIn;
    dW[end-1] = deltas[len-1] * fwd.A[len-1].Transpose();
    dB[end-1] = deltas[len-1];

    // propagate backward through rest of slice
    for (int li = static_cast<int>(len) - 2; li >= 0; --li) {
        size_t gl = start + li;  // global layer index
        DynamicMatrix err = mLayers[gl+1].weights.Transpose() * deltas[li+1];
        switch (mLayers[gl].activation) {
            case Activation::Sigmoid:
                deltas[li] = err.ElemWiseMult(fwd.A[li+1].Apply(sigmoidPrime)); break;
            case Activation::ReLU:
                deltas[li] = err.ElemWiseMult(fwd.Z[li].Apply(reluPrime));      break;
            case Activation::Tanh:
                deltas[li] = err.ElemWiseMult(fwd.A[li+1].Apply(tanhPrime));    break;
            case Activation::Softmax:
                deltas[li] = softmaxDelta(fwd.A[li+1], err);                       break;
            case Activation::Input: break;
        }
        dW[gl] = deltas[li] * fwd.A[li].Transpose();
        dB[gl] = deltas[li];
    }

    // delta at the slice input = W[start]^T * deltas[0]
    return mLayers[start].weights.Transpose() * deltas[0];
}


// ---- training step ----
float TwoHeadedNetwork::TrainStep(const DynamicMatrix& input,
                                   const DynamicMatrix& policyTarget,
                                   const DynamicMatrix& valueTarget,
                                   float lr,
                                   float policyWeight,
                                   float valueWeight)
{
    constexpr float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const size_t L = mLayers.size();
    const size_t pStart = mTrunkLen;
    const size_t vStart = mTrunkLen + mPolicyLen;
    const size_t vEnd   = vStart + mValueLen;

    // === FORWARD ===
    auto trunkFwd  = forwardSlice(0, mTrunkLen, input);
    const DynamicMatrix& trunkOut = trunkFwd.A.back();
    auto policyFwd = forwardSlice(pStart, pStart + mPolicyLen, trunkOut);
    auto valueFwd  = forwardSlice(vStart, vEnd,                trunkOut);

    const DynamicMatrix& policyOut = policyFwd.A.back();
    const DynamicMatrix& valueOut  = valueFwd.A.back();

    // === LOSSES ===
    // Policy: cross-entropy
    float policyLoss = 0.0f;
    for (size_t i = 0; i < policyOut.Rows(); i++) {
        float a = std::max(policyOut.at(i,0), 1e-7f);
        policyLoss -= policyTarget.at(i,0) * std::log(a);
    }
    // Value: MSE
    float valueLoss = 0.0f;
    for (size_t i = 0; i < valueOut.Rows(); i++) {
        float diff = valueOut.at(i,0) - valueTarget.at(i,0);
        valueLoss += diff * diff;
    }
    valueLoss *= 0.5f;

    float totalLoss = policyWeight * policyLoss + valueWeight * valueLoss;

    // === BACKWARD ===
    std::vector<DynamicMatrix> dW(L, DynamicMatrix(1,1));
    std::vector<DynamicMatrix> dB(L, DynamicMatrix(1,1));

    // Policy head output delta: cross-entropy + softmax → a - y
    DynamicMatrix policyDelta = policyOut - policyTarget;
    policyDelta = policyDelta * policyWeight;
    dW[pStart] = policyDelta * policyFwd.A[0].Transpose();
    dB[pStart] = policyDelta;
    DynamicMatrix policyGradAtTrunk = mLayers[pStart].weights.Transpose() * policyDelta;

    // Value head output delta: MSE + tanh → (a - y) * tanh'(a)
    DynamicMatrix valueDelta(valueOut.Rows(), 1);
    for (size_t i = 0; i < valueOut.Rows(); i++)
        valueDelta.at(i,0) = (valueOut.at(i,0) - valueTarget.at(i,0)) * tanhPrime(valueOut.at(i,0));
    valueDelta = valueDelta * valueWeight;
    dW[vStart] = valueDelta * valueFwd.A[0].Transpose();
    dB[vStart] = valueDelta;
    DynamicMatrix valueGradAtTrunk = mLayers[vStart].weights.Transpose() * valueDelta;

    // Sum gradients from both heads at trunk output
    DynamicMatrix trunkOutDelta = policyGradAtTrunk + valueGradAtTrunk;

    // Backward through trunk
    if (mTrunkLen > 0) {
        std::vector<DynamicMatrix> trunkDeltas(mTrunkLen, DynamicMatrix(1,1));
        // last trunk layer
        switch (mLayers[mTrunkLen-1].activation) {
            case Activation::ReLU:
                trunkDeltas[mTrunkLen-1] = trunkOutDelta.ElemWiseMult(
                    trunkFwd.Z[mTrunkLen-1].Apply(reluPrime)); break;
            case Activation::Sigmoid:
                trunkDeltas[mTrunkLen-1] = trunkOutDelta.ElemWiseMult(
                    trunkFwd.A[mTrunkLen].Apply(sigmoidPrime)); break;
            case Activation::Tanh:
                trunkDeltas[mTrunkLen-1] = trunkOutDelta.ElemWiseMult(
                    trunkFwd.A[mTrunkLen].Apply(tanhPrime)); break;
            default: trunkDeltas[mTrunkLen-1] = trunkOutDelta; break;
        }
        dW[mTrunkLen-1] = trunkDeltas[mTrunkLen-1] * trunkFwd.A[mTrunkLen-1].Transpose();
        dB[mTrunkLen-1] = trunkDeltas[mTrunkLen-1];

        for (int l = static_cast<int>(mTrunkLen) - 2; l >= 0; --l) {
            DynamicMatrix err = mLayers[l+1].weights.Transpose() * trunkDeltas[l+1];
            switch (mLayers[l].activation) {
                case Activation::ReLU:
                    trunkDeltas[l] = err.ElemWiseMult(trunkFwd.Z[l].Apply(reluPrime)); break;
                case Activation::Sigmoid:
                    trunkDeltas[l] = err.ElemWiseMult(trunkFwd.A[l+1].Apply(sigmoidPrime)); break;
                case Activation::Tanh:
                    trunkDeltas[l] = err.ElemWiseMult(trunkFwd.A[l+1].Apply(tanhPrime)); break;
                default: trunkDeltas[l] = err; break;
            }
            dW[l] = trunkDeltas[l] * trunkFwd.A[l].Transpose();
            dB[l] = trunkDeltas[l];
        }
    }

    // === ADAM UPDATE ===
    mT++;
    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(mT));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(mT));

    for (size_t l = 0; l < L; l++) {
        // skip head layers whose gradients weren't computed (policy vs value)
        // (both heads always get gradients in our scheme, so this is always valid)
        for (size_t r = 0; r < mLayers[l].weights.Rows(); r++)
            for (size_t c = 0; c < mLayers[l].weights.Cols(); c++) {
                float g = dW[l].at(r, c);
                float m = beta1 * mMW[l].at(r,c) + (1.0f - beta1) * g;
                float v = beta2 * mVW[l].at(r,c) + (1.0f - beta2) * g * g;
                mMW[l].at(r,c) = m; mVW[l].at(r,c) = v;
                mLayers[l].weights.at(r,c) -= lr * (m/bc1) / (std::sqrt(v/bc2) + eps);
            }
        for (size_t r = 0; r < mLayers[l].biases.Rows(); r++) {
            float g = dB[l].at(r,0);
            float m = beta1 * mMB[l].at(r,0) + (1.0f - beta1) * g;
            float v = beta2 * mVB[l].at(r,0) + (1.0f - beta2) * g * g;
            mMB[l].at(r,0) = m; mVB[l].at(r,0) = v;
            mLayers[l].biases.at(r,0) -= lr * (m/bc1) / (std::sqrt(v/bc2) + eps);
        }
    }

    return totalLoss;
}


// ---- save / load ----
void TwoHeadedNetwork::Save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    // architecture metadata
    uint32_t tl = mTrunkLen, pl = mPolicyLen, vl = mValueLen;
    f.write(reinterpret_cast<const char*>(&tl), sizeof(tl));
    f.write(reinterpret_cast<const char*>(&pl), sizeof(pl));
    f.write(reinterpret_cast<const char*>(&vl), sizeof(vl));

    uint32_t numLayers = static_cast<uint32_t>(mLayers.size());
    f.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));

    for (const auto& layer : mLayers) {
        uint8_t act = static_cast<uint8_t>(layer.activation);
        f.write(reinterpret_cast<const char*>(&act), sizeof(act));

        uint32_t wr = layer.weights.Rows(), wc = layer.weights.Cols();
        f.write(reinterpret_cast<const char*>(&wr), sizeof(wr));
        f.write(reinterpret_cast<const char*>(&wc), sizeof(wc));
        for (size_t r = 0; r < wr; r++)
            for (size_t c = 0; c < wc; c++) {
                float v = layer.weights.at(r,c);
                f.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }

        uint32_t br = layer.biases.Rows();
        f.write(reinterpret_cast<const char*>(&br), sizeof(br));
        for (size_t r = 0; r < br; r++) {
            float v = layer.biases.at(r,0);
            f.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    }
}

void TwoHeadedNetwork::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for reading: " + path);

    uint32_t tl, pl, vl;
    f.read(reinterpret_cast<char*>(&tl), sizeof(tl));
    f.read(reinterpret_cast<char*>(&pl), sizeof(pl));
    f.read(reinterpret_cast<char*>(&vl), sizeof(vl));
    mTrunkLen = tl; mPolicyLen = pl; mValueLen = vl;

    uint32_t numLayers;
    f.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

    mLayers.clear();
    for (uint32_t i = 0; i < numLayers; i++) {
        uint8_t act;
        f.read(reinterpret_cast<char*>(&act), sizeof(act));

        uint32_t wr, wc;
        f.read(reinterpret_cast<char*>(&wr), sizeof(wr));
        f.read(reinterpret_cast<char*>(&wc), sizeof(wc));
        DynamicMatrix weights(wr, wc);
        for (size_t r = 0; r < wr; r++)
            for (size_t c = 0; c < wc; c++) {
                float v; f.read(reinterpret_cast<char*>(&v), sizeof(v));
                weights.at(r,c) = v;
            }

        uint32_t br;
        f.read(reinterpret_cast<char*>(&br), sizeof(br));
        DynamicMatrix biases(br, 1);
        for (size_t r = 0; r < br; r++) {
            float v; f.read(reinterpret_cast<char*>(&v), sizeof(v));
            biases.at(r,0) = v;
        }

        mLayers.push_back({ std::move(weights), std::move(biases), static_cast<Activation>(act) });
    }
    initAdamState();
}
