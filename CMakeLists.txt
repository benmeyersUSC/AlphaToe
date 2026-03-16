cmake_minimum_required(VERSION 3.20)
project(AlphaToe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- shared neural network library ---
add_library(cppNN
    cppNN/DynamicMatrix.cpp
    cppNN/NeuralNetwork.cpp
    cppNN/TwoHeadedNetwork.cpp
)
target_include_directories(cppNN PUBLIC cppNN)

# --- core TTT game library ---
add_library(tttcore
    src/game/GameState.cpp
    src/game/GameRules.cpp
    src/ai/Minimax.cpp
)
target_include_directories(tttcore PUBLIC src)
target_link_libraries(tttcore PRIVATE cppNN)

# --- AlphaToe player (MCTS + NN) ---
add_library(alphatoe
    src/ai/MCTSNode.cpp
    src/ai/AlphaToePlayer.cpp
)
target_include_directories(alphatoe PUBLIC src)
target_link_libraries(alphatoe PUBLIC tttcore cppNN)

# --- training executable ---
add_executable(alphatoe_train
    src/train/main_train.cpp
    src/train/SelfPlay.cpp
    src/train/TrainingLoop.cpp
)
target_link_libraries(alphatoe_train PRIVATE alphatoe tttcore cppNN)

# --- interactive game (terminal) ---
add_executable(alphatoe_game
    src/ui/main_game.cpp
    src/ui/TerminalUI.cpp
)
target_link_libraries(alphatoe_game PRIVATE alphatoe tttcore cppNN)
