#include "GameRules.h"

// all winning triples (indices)
static const int winLines[8][3] = {
    {0,1,2}, {3,4,5}, {6,7,8},  // rows
    {0,3,6}, {1,4,7}, {2,5,8},  // cols
    {0,4,8}, {2,4,6}            // diags
};

GameResult GameRules::result(const GameState& s) {
    for (const auto& line : winLines) {
        Cell a = s.at(line[0]), b = s.at(line[1]), c = s.at(line[2]);
        if (a != Cell::Empty && a == b && b == c)
        {
            return a == Cell::X ? GameResult::XWins : GameResult::OWins;
        }
    }
    for (int i = 0; i < 9; i++)
    {
        if (s.at(i) == Cell::Empty) 
        {
            return GameResult::Ongoing;
        }
    }
    return GameResult::Draw;
}

bool GameRules::isTerminal(const GameState& s) {
    return result(s) != GameResult::Ongoing;
}

float GameRules::score(const GameState& s, Player p) {
    GameResult r = result(s);
    if (r == GameResult::Draw)    return 0.0f;
    if (r == GameResult::XWins)   return p == Player::X ?  1.0f : -1.0f;
    if (r == GameResult::OWins)   return p == Player::O ?  1.0f : -1.0f;
    return 0.0f; 
}
