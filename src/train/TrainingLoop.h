#pragma once
#include "ReplayBuffer.h"
#include "ai/AlphaToe.h"

class TrainingLoop {
public:
    // Each epoch: sample batchSize examples from the buffer (prioritized),
    // do one gradient step per example, update priorities with the new loss.
    // Returns average loss of the final epoch.
    static float run(AlphaToe& ai,
                     ReplayBuffer& buffer,
                     size_t batchSize   = 256,
                     int    epochs      = 10,
                     float  lr          = 0.001f,
                     float  policyWeight = 1.0f,
                     float  valueWeight  = 1.0f);
};
