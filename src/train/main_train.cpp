#include "SelfPlay.h"
#include "TrainingLoop.h"
#include "ReplayBuffer.h"
#include <iostream>
#include <string>

int main() {
    const int    rounds        = 10;
    const int    gamesPerRound = 50;
    const int    simulations   = 270;
    const int    epochs        = 10;
    const size_t batchSize     = 256;
    const float  lr            = 0.001f;
    const std::string bufferPath = "../data/replay_buffer.bin";

    std::cout << "AlphaToe trainer\n";
    std::cout << "  rounds=" << rounds << "  games/round=" << gamesPerRound
              << "  sims/move=" << simulations << "  epochs/round=" << epochs
              << "  batch=" << batchSize << "  lr=" << lr << "\n\n";

    AlphaToe ai;

    ReplayBuffer buffer(10000, 0.6f);
    buffer.load(bufferPath);
    std::cout << "Replay buffer: " << buffer.size() << " samples loaded\n\n";

    for (int r = 0; r < rounds; r++) {
        std::cout << "=== Round " << (r + 1) << " / " << rounds << " ===\n";

        // self-play → add to buffer → persist
        std::cout << "[Self-play]\n";
        auto newSamples = SelfPlay::run(ai, gamesPerRound, simulations);
        buffer.add(newSamples);
        buffer.save(bufferPath);
        std::cout << "  buffer size: " << buffer.size() << " samples\n\n";

        // train from prioritized buffer
        std::cout << "[Training]\n";
        float loss = TrainingLoop::run(ai, buffer, batchSize, epochs, lr);
        std::cout << "  final avg loss: " << loss << "\n\n";

        // model save omitted — call ai.save(path) explicitly when ready to persist
    }

    std::cout << "Done.\n";
    return 0;
}
