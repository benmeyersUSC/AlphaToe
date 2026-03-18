import tttgame as game
import tttgamedata as gdata
import tttgameai as ai
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random

def one_hot(total, targ):
    l = [0.0] * total
    l[targ] = 1.0
    return l

def self_play(games=100):
    # we save training data
    training_data = []
    for gm in range(games):
        tttg = game.TicTacToeGame()
        data = {
            "X":[], "O":[]
        }

        playerX = gm%2 == 0 
        if gm%27 == 0:
            playerX = not playerX
            tttg.setSquareState(random.randint(0, 8), "X")
        
        # random 1-2 moves start  
        elif gm % 50 == 0:
            num_moves = random.randint(1, 2)
            current_player = "X"
            for move_num in range(num_moves):
                available = [i for i in range(9) if tttg.getBoard().state[i] == " "]
                if available:
                    tttg.setSquareState(random.choice(available), current_player)
                    current_player = "O" if current_player == "X" else "X"
            playerX = (current_player == "X")
        
        # specific tactical positions
        elif gm % 75 == 0:
            interesting_starts = [
                [4],                    
                [0, 8],                
                [4, 0],                
                [0, 4, 8],            
            ]
            start_moves = random.choice(interesting_starts)
            current_player = "X"
            
            for pos in start_moves:
                tttg.setSquareState(pos, current_player)
                current_player = "O" if current_player == "X" else "X"
            
            playerX = (current_player == "X")

        while ai.winner(tttg.getBoard()) == " ": 
            sym = "X" if playerX else "O"
            brd = tttg.getBoard()
            if not playerX:
                new_brd = []
                for i in range(len(brd.state)):
                    if brd.state[i] == " ":
                        new_brd.append(" ")
                    else:
                        new_brd.append("X" if brd.state[i] == "O" else "O")
                brd = new_brd
            inference = ai.pickMove(brd, tttg.toe)
            move = inference[0]
            move_data = inference[1]
            data[sym].append(move_data)
            tttg.setSquareState(move, sym)

            playerX = not playerX
        
        win = ai.winner(tttg.getBoard())
        print(f"WINNER_{gm} = {win}")
        score = ai.GetScore(tttg.getBoard())
        print(f"SCORE_{gm} = {score}")

        for row in data["O"]:
            row["final_outcome"] = score
            training_data.append(row)
        for row in data["X"]:
            row["final_outcome"] = -score
            training_data.append(row)
    

    i = 1
    with open("data.csv", "a") as fn:
        for row in training_data:
            print(f"WRITING MOVE {i}")
            for sq in row["board_state"]:
                fn.write(f"{sq},")
            for p in row["mcts_policy"]:
                fn.write(f"{p},")
            fn.write(str(row["final_outcome"]))
            fn.write("\n")
            i += 1

def main():
    tttg = game.TicTacToeGame()
    # training_data = []

    while ai.winner(tttg.getBoard()) == " ":
        print("Board is now (after O):")
        tttg.printBoard()
        
        userChc = int(input("Pick a square (1-9) on which to place an X: ")) - 1
        
        tttg.setSquareState(userChc, "X")

        print("Board is now (after X):")
        tttg.printBoard()

        if ai.winner(tttg.getBoard()) == " ":
            inference = ai.pickMove(tttg.getBoard(), "O")
            move = inference[0]
            data = inference[1]
            # training_data.append(data)
            tttg.setSquareState(move, "O")
    
    
    win = ai.winner(tttg.getBoard())

    if win == "X":
        print("YOU WIN!!!")
    elif win == "O":
        print("COMPUTER WINS!!!")
    else:
        print("TIE!!!")
    tttg.printBoard()

     # after game ends, update all rows:
    # for row in training_data:
    #     row['final_outcome'] = ai.GetScore(tttg.getBoard())  


    # i = 1
    # with open("data.csv", "a") as fn:
    #     for row in training_data:
    #         print(f"WRITING MOVE {i}")
    #         for sq in row["board_state"]:
    #             fn.write(f"{sq},")
    #         for p in row["mcts_policy"]:
    #             fn.write(f"{p},")
    #         fn.write(str(row["final_outcome"]))
    #         fn.write("\n")
    #         i += 1

def get_boards(games: int, blanks=False) -> list[tuple[gdata.GameState, bool]]:
    """
    returns a tuple of game-starts of the following form:
        board (GameState), playerX (bool)
    """
    boards = []
    for i in range(games):
        if (2 if blanks else i) % 2 == 0:
            blank_board = gdata.GameState(gen=False)
            blank_board.state = [" "] * 9
            boards.append((blank_board, i%2==0))
            print(f"{i%2==0} -> {blank_board.state}")

        
        # filled in boards
        else:
            # random
            if i % 4 == 1:
                board, final_playerX = generate_random_position()
                boards.append((board, final_playerX))
            # minimax
            else:  
                # board, final_playerX = generate_minimax_position()
                # boards.append((board, final_playerX))
                blank_board = gdata.GameState(gen=False)
                blank_board.state = [" "] * 9
                boards.append((blank_board, i%3==0))
            
            print(f"{final_playerX} -> {board.state}")
    
    return boards

def generate_random_position() -> tuple[gdata.GameState, bool]:
    """randomly fill a board"""

    # dont want ended games
    num_moves = random.randint(1, 7)
    
    board = gdata.GameState()
    current_player_is_X = random.random() > 0.5
    
    for move_num in range(num_moves):
        available_squares = [i for i in range(9) if board[i] == " "]
        
        if not available_squares:
            break
            
        square = random.choice(available_squares)
        
        piece = "X" if current_player_is_X else "O"
        board[square] = piece
        
        if ai.winner(board) != " ":
            break
            
        current_player_is_X = not current_player_is_X
    
    return board, current_player_is_X

def generate_minimax_position() -> tuple[gdata.GameState, bool]:
    """fill a board with N minimax moves"""
    
    num_moves = random.randint(1, 7)
    current_player_is_X = random.random() > 0.5
    board = gdata.GameState()
    
    for move_num in range(num_moves):
        if ai.winner(board) != " ":
            break
            
        try:
            current_player_symbol = "X" if current_player_is_X else "O"
            minimax_move = ai.pickMove(board, None, current_player_symbol)
            if isinstance(minimax_move, tuple):
                minimax_move = minimax_move[0]
        except Exception as e:
            available_squares = [i for i in range(9) if board[i] == " "]
            if not available_squares:
                break
            minimax_move = random.choice(available_squares)
        
        piece = "X" if current_player_is_X else "O"
        board[minimax_move] = piece
        
        current_player_is_X = not current_player_is_X

    return board, current_player_is_X

def add_move(tttg : game.TicTacToeGame, moveStr : str) -> game.TicTacToeGame:
        if ai.winner(tttg.currentState) != " ":
            return tttg
        if moveStr != "X" and moveStr != "O":
            return tttg
        
        available = [i for i, v in enumerate(tttg.currentState) if v == " "]
        x = random.choice(available)

        tttg.currentState[x] = moveStr

        return tttg

def alphazero_training_loop(
                            iterations=5, 
                            games_per_iteration=200, 
                            training_epochs=50, 
                            batch_size=32, 
                            learning_rate=0.001,
                            csv_file="data.csv",
                            plot_not_play=True,
                            blanks=False
                           ):
    """
    Play, Train, Repeat
    """
    
    print(f"Iterations: {iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"Training epochs per iteration: {training_epochs}")
    print("-" * 60)
    
    # error and data size over time
    iteration_losses = []
    data_sizes = []

    # single TTTGame, single AlphaToe within (being trained!)
    master_game = game.TicTacToeGame()
    
    for iteration in range(iterations):
        print(f"\n{'='*20} ITERATION {iteration + 1}/{iterations} {'='*20}")
        
        # 1: TRAINING
        print(f"\nTraining AlphaToe...")

        # load data
        try:
            data_df = pd.read_csv(csv_file, header=0, dtype=float)
            if data_df.isnull().any().any():
                data_df = data_df.dropna()

            MAX_EXAMPLES = 5000
            if len(data_df) > MAX_EXAMPLES:
                data_df = data_df.iloc[-MAX_EXAMPLES:]

                        
            print(f"{len(data_df)} rows...")
            data_sizes.append(len(data_df))
            
            # slice rows
            board_states = data_df.iloc[:, 0:9].values.astype(np.float32)
            policy_targets = data_df.iloc[:, 9:18].values.astype(np.float32)
            value_targets = data_df.iloc[:, 18].values.astype(np.float32)
            
            # put segments into torch tensors
            X = torch.FloatTensor(board_states)
            policy_y = torch.FloatTensor(policy_targets)
            value_y = torch.FloatTensor(value_targets).unsqueeze(1)
            
            # torch dataset and dataloader
            dataset = TensorDataset(X, policy_y, value_y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # torch train AlphaToe
            master_game.toe.net.train()
            
            optimizer = optim.Adam(master_game.toe.net.parameters(), lr=learning_rate)
            # policy_loss_fn = nn.CrossEntropyLoss()
            policy_loss_fn = nn.MSELoss()
            value_loss_fn = nn.MSELoss()
            
            epoch_losses = []
            
            for epoch in range(training_epochs):
                epoch_total_loss = 0.0
                num_batches = 0
                
                for batch_boards, batch_policies, batch_values in dataloader:
                    # clear gradients
                    optimizer.zero_grad()
                    
                    # feed forward
                    policy_logits, value_pred = master_game.toe.net(batch_boards)
                    
                    # calculate losses
                    policy_loss = policy_loss_fn(policy_logits, batch_policies)
                    value_loss = value_loss_fn(value_pred, batch_values)
                    total_loss = policy_loss + value_loss
                    
                    # calculate gradient
                    total_loss.backward()
                    # make changes
                    optimizer.step()
                    
                    # accumulate losses for average calc
                    epoch_total_loss += total_loss.item()
                    num_batches += 1
                
                avg_loss = epoch_total_loss / num_batches
                epoch_losses.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{training_epochs}: Loss = {avg_loss:.4f}")
            
            iteration_losses.append(epoch_losses[-1])
            print(f"Final loss: {epoch_losses[-1]:.4f}")
            
            master_game.toe.net.eval()
            
        except Exception as e:
            print(f"BAD: {e}")

        
        # 2: SELF PLAY
        print(f"\nSelf-playing {games_per_iteration} games...")
        
        training_data = []
        wins = {"X": 0, "O": 0, "Tie": 0}
        
        game_count = 0
        # (GameState, playerX)
        boards = get_boards(games_per_iteration, blanks)
        # for gm in boards:
        #     tttg = game.TicTacToeGame()
        #     # copy the master trained network to this game
        #     tttg.toe.net.load_state_dict(master_game.toe.net.state_dict())
        #     tttg.toe.net.eval()

        #     game_data = {"X": [], "O": []}
        #     # extract player
        #     playerX = gm[1]
        #     # extract game state
        #     tttg.currentState = gm[0]

            

        #     if random.random() > 0.7:
        #         tttg = add_move(tttg, "X" if playerX else "O")
        #         # tttg.currentState[random.choice(range(9))] = "X" if playerX else "O"
        #         playerX = not playerX
        #         print(f"BOARD NOW: {tttg.currentState.state} on {"X" if playerX else "O"}")
        #     if random.random() > 0.8:
        #         tttg = add_move(tttg, "X" if playerX else "O")
        #         # tttg.currentState[random.choice(range(9))] = "X" if playerX else "O"
        #         playerX = not playerX
        #         print(f"BOARD NOW: {tttg.currentState.state} on {"X" if playerX else "O"}")
        #     if random.random() > 0.9:
        #         tttg = add_move(tttg, "X" if playerX else "O")
        #         # tttg.currentState[random.choice(range(9))] = "X" if playerX else "O"
        #         playerX = not playerX
        #         print(f"BOARD NOW: {tttg.currentState.state} on {"X" if playerX else "O"}")

        #     while ai.winner(tttg.getBoard()) == " ":
        #         current_player = "X" if playerX else "O"
        #         brd = tttg.getBoard()
                
        #         inference = ai.pickMove(brd, tttg.toe, current_player)
        #         move = inference[0]
        #         move_data = inference[1]
        #         game_data[current_player].append(move_data)

        #         # mmax_results = ai.pickMove(brd, None, current_player)
        #         # mmax_data = ai.get_training_data_perspective(
        #         #             brd, 
        #         #             current_player, 
        #         #             one_hot(9, mmax_results[0]), 
        #         #             None 
        #         #         )
        #         # game_data[current_player].append(mmax_data)
                
        #         tttg.setSquareState(move, current_player)
                
        #         playerX = not playerX

        #     win = ai.winner(tttg.getBoard())
        #     if win == "N":
        #         wins["Tie"] += 1
        #     else:
        #         wins[win] += 1

        #     score = ai.GetScore(tttg.getBoard()) 

        #     for row in game_data["X"]:
        #         if score == 1:      # O won
        #             row["final_outcome"] = -1.0
        #         elif score == -1:   # X won  
        #             row["final_outcome"] = 1.0
        #         else:               # Tie
        #             row["final_outcome"] = 0.0
        #         training_data.append(row)
                
        #     for row in game_data["O"]:
        #         if score == 1:      # O won
        #             row["final_outcome"] = 1.0
        #         elif score == -1:   # X won
        #             row["final_outcome"] = -1.0  
        #         else:               # Tie
        #             row["final_outcome"] = 0.0
        #         training_data.append(row)

        #     if (game_count + 1) % 50 == 0:
        #         print(f"   Completed {game_count + 1}/{games_per_iteration} games")
        #     game_count += 1

        # print(f"Game results: X: {wins['X']}, O: {wins['O']}, Tie: {wins['Tie']}")


        # if iteration == 0:  # Only print on first iteration to avoid spam
            # print_board_distribution_stats(boards)

        for gm in boards:
            tttg = game.TicTacToeGame()
            # copy the master trained network to this game
            tttg.toe.net.load_state_dict(master_game.toe.net.state_dict())
            tttg.toe.net.eval()

            game_data = {"X": [], "O": []}
            # extract player
            playerX = gm[1]
            # extract game state
            tttg.currentState = gm[0]

            if random.random() > 0.7:
                tttg = add_move(tttg, "X" if playerX else "O")
                playerX = not playerX

            while ai.winner(tttg.getBoard()) == " ":
                current_player = "X" if playerX else "O"
                brd = tttg.getBoard()
                
                inference = ai.pickMove(brd, tttg.toe, current_player)
                move = inference[0]
                move_data = inference[1]
                game_data[current_player].append(move_data)
                
                tttg.setSquareState(move, current_player)
                playerX = not playerX

            win = ai.winner(tttg.getBoard())
            if win == "N":
                wins["Tie"] += 1
            else:
                wins[win] += 1

            score = ai.GetScore(tttg.getBoard()) 

            # rest of the outcome assignment stays the same...
            for row in game_data["X"]:
                if score == 1:      # O won
                    row["final_outcome"] = -1.0
                elif score == -1:   # X won  
                    row["final_outcome"] = 1.0
                else:               # Tie
                    row["final_outcome"] = 0.0
                training_data.append(row)
                
            for row in game_data["O"]:
                if score == 1:      # O won
                    row["final_outcome"] = 1.0
                elif score == -1:   # X won
                    row["final_outcome"] = -1.0  
                else:               # Tie
                    row["final_outcome"] = 0.0
                training_data.append(row)

            if (game_count + 1) % 50 == 0:
                print(f"   Completed {game_count + 1}/{games_per_iteration} games")
            game_count += 1

        print(f"Game results: X: {wins['X']}, O: {wins['O']}, Tie: {wins['Tie']}")

        # 3: SAVE NEW DATA
        print(f"\n3: Saving {len(training_data)} new training examples...")

        with open(csv_file, "a") as fn:
            for row in training_data:
                for field in ["board_state", "mcts_policy"]:
                    for record in row[field]:
                        fn.write(f"{record},")
                fn.write(str(row["final_outcome"]))
                fn.write("\n")
        
        print(f"Win rates: X: {wins['X']/games_per_iteration}, O: {wins['O']/games_per_iteration}, Tie: {wins['Tie']/games_per_iteration}")
    
    if iterations >0:
        # FINAL SUMMARY
        print(f"\n{'='*20} TRAINING COMPLETE {'='*20}")

        print(f"Final dataset size: {data_sizes[-1] if data_sizes else 0} moves")
        print(f"Training loss improvement: {iteration_losses[0]:.4f} --> {iteration_losses[-1]:.4f}")

    # plot loop stats
    if plot_not_play:
        if len([x for x in iteration_losses if x is not None]) > 1:
            plt.figure(figsize=(12, 9))
            
            plt.subplot(1, 2, 1)
            losses = [(i+1, loss) for i, loss in enumerate(iteration_losses) if loss is not None]
            if losses:
                iterations_with_data, losses = zip(*losses)
                plt.plot(iterations_with_data, losses, 'bo-')
                plt.title('Training Loss Over Iterations')
                plt.xlabel('Iteration')
                plt.ylabel('Final Training Loss')
                plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(data_sizes)+1), data_sizes, 'go-')
            plt.title('Dataset Size Growth')
            plt.xlabel('Iteration')
            plt.ylabel('Training Examples')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
    else:
        while input("Would you like to play against AlphaToe? (Q to quit):").upper() != "Q":
            tttg = game.TicTacToeGame()
            # copy the master trained network to this game
            tttg.toe.net.load_state_dict(master_game.toe.net.state_dict())
            tttg.toe.net.eval()
            while ai.winner(tttg.getBoard()) == " ":
                print("Board is now (after O):")
                tttg.printBoard()
                
                userChc = int(input("Pick a square (1-9) on which to place an X: ")) - 1
                
                tttg.setSquareState(userChc, "X")

                print("Board is now (after X):")
                tttg.printBoard()

                if ai.winner(tttg.getBoard()) == " ":
                    inference = ai.pickMove(tttg.getBoard(), tttg.toe, "O")
                    # inference = ai.pickMove(tttg.getBoard(), None, "O")
                    move = inference[0]
                    data = inference[1]
                    # training_data.append(data)
                    tttg.setSquareState(move, "O")
            
            
            win = ai.winner(tttg.getBoard())

            if win == "X":
                print("YOU WIN!!!")
            elif win == "O":
                print("COMPUTER WINS!!!")
            else:
                print("TIE!!!")
            tttg.printBoard()

    return iteration_losses, data_sizes



if __name__ == "__main__":
    alphazero_training_loop(
    iterations=0,       # just play!           
    games_per_iteration=1, 
    training_epochs=270,     
    batch_size=128,          
    learning_rate=0.0003,    
    plot_not_play=False,
    blanks=True)

