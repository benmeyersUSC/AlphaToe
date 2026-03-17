#pragma once
#include "SelfPlay.h"
#include <vector>
#include <string>
#include <random>

struct PrioritizedSample {
    GameSample sample;
    float priority;
};

// Fixed-capacity replay buffer with prioritized sampling.
//
// New samples enter with the current max priority so they are
// sampled at least once before any priority update.
//
// Sampling probability ∝ priority^alpha.
// After each gradient step, call updatePriority(idx, loss) to
// keep priorities current — this is what drives the prioritization.
//
// When the buffer exceeds maxSize, the oldest samples are evicted (FIFO).
class ReplayBuffer {
    std::vector<PrioritizedSample> mBuffer;
    size_t mMaxSize;
    float  mAlpha;        // prioritization exponent (0 = uniform, 1 = full)
    float  mMaxPriority;  // new samples always get this so they're sampled soon

public:
    explicit ReplayBuffer(size_t maxSize = 10000, float alpha = 0.6f)
        : mMaxSize(maxSize), mAlpha(alpha), mMaxPriority(1.0f) {}

    // Add new samples. New samples get mMaxPriority.
    // Evicts oldest entries if over capacity.
    void add(const std::vector<GameSample>& samples);

    // Sample n indices weighted by priority^alpha (with replacement).
    std::vector<size_t> sample(size_t n, std::mt19937& rng) const;

    // Update priority for a sample after a training step.
    // loss is the per-sample combined loss.
    void updatePriority(size_t idx, float loss);

    size_t size()                             const { return mBuffer.size(); }
    const PrioritizedSample& operator[](size_t i) const { return mBuffer[i]; }

    void save(const std::string& path) const;
    void load(const std::string& path);
};
