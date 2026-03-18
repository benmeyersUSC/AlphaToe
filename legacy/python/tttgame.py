import tttgameai as ai
import tttgamedata as data

class TicTacToeGame:
    # Function: Constructor
    # Purpose: Creates a GameState with empty spots
    # Input: None
    # Returns: None
    def __init__(self):
        # Represents the current game state
        # Update this variable as the game progresses
        self.currentState = data.GameState()
        self.toe = ai.AlphaToe()
    
    # Function: getBoard
    # Purpose: Return the current GameState
    # Input: None
    # Returns: The current game state
    def getBoard(self) -> data.GameState:
        return self.currentState

    # Function: setSquareState
    # Purpose: Claims the entered state for the inputted player
    # Input:    1. Unsigned short for row
    #           2. Unsigned short for column
    #           3. The player's symbol (X or O)
    # Returns: boolean -- true if the spot was set successfully
    # Note: Uses the row/column to identify a board position
    def setSquareState(self, spot: int, state: str) -> bool:
        val = False
        if spot is not None and self.currentState[spot] == " ":
            self.currentState[spot] = state
            val = True
        return val

    # Function: getWinner
    # Purpose: Determines if there's a game winner
    # Input: None
    # Returns: Character to determine winner
    #           'X' to indicate X player wins
    #           'O' to indicate O player wins
    #           'N' to indicate tie game
    #           ' ' to indicate no winner yet
    def getWinner(self) -> str:
        return ai.winner(self.currentState)
    
    def printBoard(self):
        print(" ", end="")
        print("-" * 11)
        for row in range(3):
            print("|", end="")
            for col in range(3):
                idx = row * 3 + col
                cell = self.currentState[idx]
                display = cell if cell != " " else str(idx + 1)
                print(f" {display} |", end="")
            print()
            print(" ", end="")
            print("-" * 11)