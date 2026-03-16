#include "TerminalUI.h"
#include <iostream>
#include <string>

static char cellChar(Cell c) {
    switch (c) {
        case Cell::X:     return 'X';
        case Cell::O:     return 'O';
        case Cell::Empty: return '.';
    }
    return '?';
}

void TerminalUI::printBoard(const GameState& s) {
    std::cout << "\n";
    for (int row = 0; row < 3; row++) {
        std::cout << " ";
        for (int col = 0; col < 3; col++) {
            int sq = row * 3 + col;
            char c = cellChar(s.at(sq));
            // show square number if empty, piece otherwise
            if (c == '.') std::cout << (sq + 1);
            else          std::cout << c;
            if (col < 2) std::cout << " | ";
        }
        std::cout << "\n";
        if (row < 2) std::cout << "---+---+---\n";
    }
    std::cout << "\n";
}

void TerminalUI::printResult(GameResult r, Player humanPlayer) {
    switch (r) {
        case GameResult::Draw:   std::cout << "DRAW\n"; break;
        case GameResult::XWins:
            std::cout << (humanPlayer == Player::X ? "You win!\n" : "You lose!\n"); break;
        case GameResult::OWins:
            std::cout << (humanPlayer == Player::O ? "You win!\n" : "You lose!\n"); break;
        default: break;
    }
}

int TerminalUI::getHumanMove(const GameState& s) {
    while (true) {
        std::cout << "Your move (1-9): ";
        int n;
        if (!(std::cin >> n)) {
            if (std::cin.eof()) return -1; // we done
            std::cin.clear();
            std::cin.ignore(1000, '\n');
            continue;
        }
        int sq = n - 1;
        if (sq < 0 || sq > 8) {
            std::cout << "Enter a number 1-9.\n";
            continue;
        }
        if (!s.empty(sq)) {
            std::cout << "Square " << n << " is taken.\n";
            continue;
        }
        return sq;
    }
}

int TerminalUI::mainMenu() {
    std::cout << "\n=== AlphaToe ===\n";
    std::cout << "  1. Play vs Minimax\n";
    std::cout << "  2. Play vs AlphaToe\n";
    std::cout << "Choice: ";
    int choice;
    while (!(std::cin >> choice) || choice < 1 || choice > 2) {
        std::cin.clear();
        std::cin.ignore(1000, '\n');
        std::cout << "Enter 1 or 2: ";
    }
    return choice;
}

Player TerminalUI::chooseSide() {
    std::cout << "Play as X or O? (X goes first) [X/O]: ";
    char c;
    while (true) {
        std::cin >> c;
        if (c == 'x' || c == 'X') return Player::X;
        if (c == 'o' || c == 'O') return Player::O;
        std::cout << "Enter x or o: ";
    }
}
