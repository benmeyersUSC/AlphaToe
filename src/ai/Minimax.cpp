#include "Minimax.h"
#include "game/GameRules.h"
#include <limits>

float Minimax::search(const GameState& s, Player rootPlayer, bool maximizing) {
    // base case
    if (GameRules::isTerminal(s))
    {
        // score with respect to rootPlayer
        return GameRules::score(s, rootPlayer);
    }

    // if maximizing, player is root, otherwise opposite
    Player mover = maximizing ? rootPlayer : opponent(rootPlayer);
    // if maximizing, we init with -inf
    float best = maximizing ? -std::numeric_limits<float>::infinity()
                            :  std::numeric_limits<float>::infinity();

    // for each available move
    for (int sq : s.legalMoves()) {
        // get value from mover marking here, and recurse now with opposite goal
        float val = search(s.apply(sq, mover), rootPlayer, !maximizing);
        best = maximizing ? std::max(best, val) : std::min(best, val);
    }
    return best;
}

int Minimax::bestMove(const GameState& s, Player player) {
    float bestVal = -std::numeric_limits<float>::infinity();
    int bestSq = -1;

    // for all available moves
    for (int sq : s.legalMoves()) {
        // get value this move if player marks here:
        //      here, we're choosing max of the minimizing plays. for this we need...
        //          min of maximizing plays for next board, which needs...
        //              max of minimizing...
        float val = search(s.apply(sq, player), player, false);
        if (val > bestVal) {
            bestVal = val;
            bestSq  = sq;
        }
    }
    return bestSq;
}
