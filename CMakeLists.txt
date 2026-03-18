cmake_minimum_required(VERSION 3.20)
project(AlphaToe CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- shared neural network library (header-only: Tensor.hpp + NeuralNet.hpp) ---
add_library(cppNN INTERFACE)
target_include_directories(cppNN INTERFACE cppNN)

# --- core TTT game library ---
add_library(tttcore
    src/game/GameState.cpp
    src/game/GameRules.cpp
    src/ai/Minimax.cpp
)
target_include_directories(tttcore PUBLIC src)

# --- AlphaToe player (MCTS + NN) ---
add_library(alphatoe
    src/ai/MCTSNode.cpp
    src/ai/AlphaToe.cpp
    src/ai/Minimax.cpp
)
target_include_directories(alphatoe PUBLIC src)
target_link_libraries(alphatoe PUBLIC tttcore cppNN)

# --- training executable ---
add_executable(alphatoe_train
    src/train/main_train.cpp
    src/train/SelfPlay.cpp
    src/train/TrainingLoop.cpp
    src/train/ReplayBuffer.cpp
)
target_link_libraries(alphatoe_train PRIVATE alphatoe tttcore cppNN)

# --- interactive game (terminal) ---
add_executable(alphatoe_game
    src/ui/main_game.cpp
    src/ui/TerminalUI.cpp
)
target_link_libraries(alphatoe_game PRIVATE alphatoe tttcore cppNN)
