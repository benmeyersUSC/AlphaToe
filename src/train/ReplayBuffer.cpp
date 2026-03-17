#include "ReplayBuffer.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <filesystem>
#include <numeric>

void ReplayBuffer::add(const std::vector<GameSample>& samples) {
    for (const auto& s : samples)
        mBuffer.push_back({ s, mMaxPriority });

    // evict oldest entries in one shot if over capacity
    if (mBuffer.size() > mMaxSize) {
        size_t excess = mBuffer.size() - mMaxSize;
        mBuffer.erase(mBuffer.begin(), mBuffer.begin() + static_cast<long>(excess));
    }
}

std::vector<size_t> ReplayBuffer::sample(size_t n, std::mt19937& rng) const {
    std::vector<float> weights(mBuffer.size());
    for (size_t i = 0; i < mBuffer.size(); i++)
        weights[i] = std::pow(mBuffer[i].priority, mAlpha);

    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());

    std::vector<size_t> indices(n);
    for (auto& idx : indices)
        idx = dist(rng);
    return indices;
}

void ReplayBuffer::updatePriority(size_t idx, float loss) {
    float p = std::abs(loss) + 1e-6f;  // small epsilon ensures non-zero priority
    mBuffer[idx].priority = p;
    mMaxPriority = std::max(mMaxPriority, p);
}

// ── binary save/load ──────────────────────────────────────────────────────────
// Format per entry: float[9] board | float[9] policy | float value | float priority

void ReplayBuffer::save(const std::string& path) const {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    uint32_t n = static_cast<uint32_t>(mBuffer.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& ps : mBuffer) {
        f.write(reinterpret_cast<const char*>(ps.sample.board.data()),  sizeof(ps.sample.board));
        f.write(reinterpret_cast<const char*>(ps.sample.policy.data()), sizeof(ps.sample.policy));
        f.write(reinterpret_cast<const char*>(&ps.sample.value),        sizeof(ps.sample.value));
        f.write(reinterpret_cast<const char*>(&ps.priority),            sizeof(ps.priority));
    }
}

void ReplayBuffer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return;  // no existing buffer — start fresh

    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));

    mBuffer.resize(n);
    for (auto& ps : mBuffer) {
        f.read(reinterpret_cast<char*>(ps.sample.board.data()),  sizeof(ps.sample.board));
        f.read(reinterpret_cast<char*>(ps.sample.policy.data()), sizeof(ps.sample.policy));
        f.read(reinterpret_cast<char*>(&ps.sample.value),        sizeof(ps.sample.value));
        f.read(reinterpret_cast<char*>(&ps.priority),            sizeof(ps.priority));
    }

    // restore mMaxPriority from loaded data
    for (const auto& ps : mBuffer)
        mMaxPriority = std::max(mMaxPriority, ps.priority);
}
