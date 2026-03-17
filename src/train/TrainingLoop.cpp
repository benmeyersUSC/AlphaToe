#include "TrainingLoop.h"
#include <random>
#include <iostream>
#include <iomanip>

float TrainingLoop::run(AlphaToe& ai,
                        ReplayBuffer& buffer,
                        size_t batchSize,
                        int epochs,
                        float lr,
                        float policyWeight,
                        float valueWeight)
{
    std::mt19937 rng(std::random_device{}());
    float finalLoss = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // clamp batch to however much data we actually have
        size_t n = std::min(batchSize, buffer.size());
        auto indices = buffer.sample(n, rng);

        float totalLoss = 0.0f;
        for (size_t idx : indices) {
            const auto& s = buffer[idx].sample;

            DynamicMatrix board(9, 1);
            for (int j = 0; j < 9; j++) board.at(j, 0) = s.board[j];

            DynamicMatrix policy(9, 1);
            for (int j = 0; j < 9; j++) policy.at(j, 0) = s.policy[j];

            DynamicMatrix value(1, 1);
            value.at(0, 0) = s.value;

            float loss = ai.trainStep(board, policy, value, lr, policyWeight, valueWeight);
            buffer.updatePriority(idx, loss);
            totalLoss += loss;
        }

        float avgLoss = totalLoss / static_cast<float>(n);
        finalLoss = avgLoss;

        std::cout << "  epoch " << std::setw(3) << (epoch + 1) << "/" << epochs
                  << "  avg loss: " << std::fixed << std::setprecision(5) << avgLoss
                  << "  (sampled " << n << " / " << buffer.size() << " in buffer)\n";
    }

    return finalLoss;
}
