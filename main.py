import tttgame as game
import tttgamedata as data
import tttgameai as ai

def main():
    tttg = game.TicTacToeGame()

    while ai.winner(tttg.getBoard()) == " ":
        print("Board is now (after O):")
        tttg.printBoard()
        
        userChc = int(input("Pick a square (1-9) on which to place an X: ")) - 1
        
        tttg.setSquareState(userChc, "X")

        print("Board is now (after X):")
        tttg.printBoard()

        if ai.winner(tttg.getBoard()) == " ":
            tttg.setSquareState(ai.pickMove(
                    tttg.getBoard(), tttg.toe
                ), "O")
    
    win = ai.winner(tttg.getBoard())

    if win == "X":
        print("YOU WIN!!!")
    elif win == "O":
        print("COMPUTER WINS!!!")
    else:
        print("TIE!!!")
    tttg.printBoard()
    

if __name__ == "__main__":
    main()