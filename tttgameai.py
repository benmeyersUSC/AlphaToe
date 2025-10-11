
import tttgamedata as data
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class GTNode:
	def __init__(self):
		self.mState = data.GameState()
		self.mChildren = []

		# MCTS
		self.visit_counts = [0] * 9
		self.value_sums = [0.0] * 9
		self.total_visits = 0
	
	def clear(self):
		self.visit_counts = [0] * 9
		self.value_sums = [0.0] * 9
		self.total_visits = 0

		for c in self.mChildren:
			c.clear()

#################################################################################################################
#################################################################################################################
def _board_to_list(board: data.GameState, playerX : bool) -> list:
	"""convert GameState object to list: X=1.0, O=-1.0, empty=0.0"""
	tensor_board = []
	for i in range(9):
		if board[i] == "X":
			tensor_board.append(1.0 if playerX else -1.0)
		elif board[i] == "O":
			tensor_board.append(-1.0 if playerX else 1.0)
		else:
			tensor_board.append(0.0)
	return tensor_board

class AlphaToe:
	def __init__(self):
		self.net = TwoHeadedNet()
		self.net.eval()  
	

	def _get_valid_moves_mask(self, board: data.GameState) -> torch.Tensor:
		"""mask where 1 = valid move, 0 = invalid move"""
		mask = []
		for i in range(9):
			mask.append(1.0 if board[i] == " " else 0.0)
		return torch.tensor(mask, dtype=torch.float32)
	
	def policy(self, board: data.GameState, player_to_move: str) -> torch.Tensor:
		"""probability distribution over moves from player_to_move's perspective"""
		board_tensor = board_to_tensor_from_perspective(board, player_to_move)
		# load mask
		valid_mask = self._get_valid_moves_mask(board)
		
		with torch.no_grad():
			# get logits
			policy_logits, _ = self.net(board_tensor)
			policy_logits = policy_logits.squeeze(0)
			
			# mask invalid moves
			policy_logits = policy_logits + (valid_mask - 1) * 1e9
			
			# softmax to get probabilities
			policy_probs = F.softmax(policy_logits, dim=0)
			
		return policy_probs
	
	def value(self, board: data.GameState, player_to_move: str) -> float:
		"""estimated value of the position from player_to_move's perspective"""
		board_tensor = board_to_tensor_from_perspective(board, player_to_move)
		
		with torch.no_grad():
			_, value = self.net(board_tensor)
			
		return value.item()

	def pickMove(self, board: data.GameState, player_to_move: str, rounds=540) -> tuple:
		"""
		Pick a move using MCTS from player_to_move's perspective
		Returns (move_index, training_data)
		"""
		root = GTNode()
		root.mState = board

		for i in range(rounds):
			path = []
			currNode = root
			current_player = player_to_move
			path.append((root, current_player))

			while len(currNode.mChildren) > 0 and winner(currNode.mState) == " ":
				currNode = self.ucb1(currNode, current_player)
				current_player = "O" if current_player == "X" else "X"
				path.append((currNode, current_player))
			
			if winner(currNode.mState) == " ":
				currNode = self.expand(currNode, current_player)
				path.append((currNode, current_player))

			if winner(currNode.mState) != " ":
				game_score = GetScore(currNode.mState)  
				if current_player == "O":
					value = game_score
				else:  # current_player == "X"
					value = -game_score
			else:
				value = self.value(currNode.mState, current_player)

			self.backup(path, value)
		
		sm = sum(root.visit_counts)
		visit_count_probabilities = [root.visit_counts[i]/(1.0 * sm) for i in range(9)]

		train_data = get_training_data_perspective(
			board, 
			player_to_move, 
			visit_count_probabilities, 
			None 
		)

		best_move = torch.argmax(torch.tensor(root.visit_counts, dtype=torch.float32)).item()
		return best_move, train_data

	def expand(self, currNode: GTNode, player_to_move: str) -> GTNode:
		"""expand node by adding all possible children"""
		valid_moves = []

		for i in range(9):
			if currNode.mState[i] == " ":
				newNode = GTNode()
				for j in range(9):
					if i == j:
						newNode.mState[j] = player_to_move
					else:
						newNode.mState[j] = currNode.mState[j]
				currNode.mChildren.append(newNode)
				valid_moves.append(i)
				
		policy_probs = self.policy(currNode.mState, player_to_move)

		best_valid_idx = 0
		best_prob = -1
		for idx, move in enumerate(valid_moves):
			if policy_probs[move] > best_prob:
				best_prob = policy_probs[move]
				best_valid_idx = idx
		
		return currNode.mChildren[best_valid_idx]

	def ucb1(self, currNode: GTNode, current_player: str) -> GTNode:
		if len(currNode.mChildren) == 0:
			raise ValueError("Needs children!")
		
		C = 1.8
		ucb_scores = []
		
		# map child indices to actual move indices (board squares)
		child_to_move = []
		for i in range(9):
			if currNode.mState[i] == " ":
				child_to_move.append(i)
		
		for child_idx in range(len(currNode.mChildren)):
			move_idx = child_to_move[child_idx]  
			visits = currNode.visit_counts[move_idx]  
			
			if visits == 0:
				ucb_scores.append(float('inf'))
			else:
				avg_val = currNode.value_sums[move_idx] / visits
				# CRITICAL FIX: Remove the value flipping entirely
				# The backup function already handles perspective correctly
				# Don't flip values here!
				
				exploration = C * math.sqrt(math.log(currNode.total_visits) / visits)
				final_score = avg_val + exploration
				ucb_scores.append(final_score)
		
		best_ind = ucb_scores.index(max(ucb_scores))
		return currNode.mChildren[best_ind]
	def backup(self, path, leaf_value):
		"""
		Backup values through the search path
		leaf_value is from the perspective of the player who made the leaf move
		"""
		# The leaf value is from the leaf player's perspective
		current_value = leaf_value
		
		# Go backwards through the path (skip the leaf itself)
		for i in range(len(path) - 1, 0, -1):
			child_node, child_player = path[i]
			parent_node, parent_player = path[i-1]
			
			# Find which move led from parent to child
			move_index = self._find_move_index(parent_node, child_node)
			
			# The value for the parent is the negative of the child's value
			# (what's good for child is bad for parent in zero-sum game)
			parent_value = -current_value
			
			# Update parent's statistics for this move
			parent_node.visit_counts[move_index] += 1
			parent_node.value_sums[move_index] += parent_value
			parent_node.total_visits += 1
			
			# For next iteration, the parent's value becomes current value
			current_value = parent_value

	def _find_move_index(self, parent, child):
		for i in range(9):
			if parent.mState[i] == " " and child.mState[i] != " ":
				return i
		raise ValueError("No move from parent gets to that child")

#################################################################################################################
#################################################################################################################


# Function: winner
# Purpose: Take in a GameState object and determine winner (or none) for several other functions
# Input: GameState object by reference
# Returns: char representing winner (X or O), tie (N), or not over yet (' ')
def winner(state : data.GameState) -> str:
	for check in ["X", "O"]:
		# rows
		for j in [0, 3, 6]:
			if (
				state[j] == check and state[j+1] == check and state[j+2] == check
			):
				return check
		# columns
		for j in [0, 1, 2]:
			if (
				state[j] == check and state[j+3] == check and state[j+6] == check
			):
				return check
		# diags
		if (
			state[0] == check and state[4] == check and state[8] == check
		):
			return check
		if (
			state[2] == check and state[4] == check and state[6] == check
		):
			return check

	for i in range(9):
		if state[i] == " ":
			return " "
		
	return "N"

# Function: GenStates
# Purpose: Generates the game tree starting at the inputted root
# Input: A GTNode pointer with the 1st state completed
#		A boolean to indicate who's turn it is (true means it's X's turn)
# Returns: Nothing
def GenStates(root : GTNode, xPlayer : bool):
	if (winner(root.mState) != " "):
		return

	# if xPlayer is true, then the symbol we're adding is X, otherwise O
	thisOne = "X" if xPlayer else "O"

	for i in range(9):
		if root.mState[i] == " ":
			# then make new node which will have a new SquareState (thisOne) at [r][c]
			newNode = GTNode()
			for j in range(9):
				# in inner loop for new board, if we're at [r][c], then make the new char
				if i == j:
					newNode.mState[j] = thisOne
				else:
					newNode.mState[j] = root.mState[j]
			# add child to root
			root.mChildren.append(newNode)
			# recursively generate children for new child!
			GenStates(newNode, not xPlayer)


# Function: GetScore
# Purpose: Examines the inputted game board to determine a winner
# Input: A game state representing a game board
# Returns: Everything is relative to O, so...
#		1.0 to indicate O wins on this board
#		0.0 to indicate tie game
#		-1.0 to indicate X wins on this board
def GetScore(state : data.GameState) -> float:
	wnr = winner(state)
	return 1.0 if wnr == "O" else -1.0 if wnr == "X" else 0.0

# Function: MinPlayer
# Purpose: Determines the minimum score this branch (or leaf) can yield
# Input: A GameTree node
# Returns: The game score meaning...
#		1.0 to indicate O wins on this board
#		0.0 to indicate tie game
#		-1.0 to indicate X wins on this board
def MinPlayer(node : GTNode) -> float:
	# if leaf, return it
	if len(node.mChildren) == 0:
		return GetScore(node.mState)

	curMin = 10.0
	for child in node.mChildren:
		curMin = min(curMin, MaxPlayer(child))
	return curMin

# Function: MaxPlayer
# Purpose: Determines the maximum score this branch (or leaf) can yield
# Input: A GameTree node
# Returns: The game score meaning...
#		1.0 to indicate O wins on this board
#		0.0 to indicate tie game
#		-1.0 to indicate X wins on this board
def MaxPlayer(node : GTNode) -> float:
	# if leaf, return it
	if len(node.mChildren) == 0:
		return GetScore(node.mState)

	curMax = -10.0
	for child in node.mChildren:
		curMax = max(curMax, MinPlayer(child))
	return curMax

# Function: MinimaxDecide
# Purpose: Determines which subtree leads to O winning
# Input: A GameTree node
# Returns: The GTNode* with the winning game state
def MinimaxDecide(root : GTNode) -> GTNode:
	# if leaf, return it
	if len(root.mChildren) == 0:
		return root
	best = None
	curMax = -10.0
	for child in root.mChildren:
		# get max score
		crit = MinPlayer(child)
		if crit > curMax:
			best = child
			curMax = crit
	return best
	
# Function: pickMove
# Purpose: Decides which square the AI should select
# Input: A game state with the current board's state
# Returns: The square number (1 through 9) the AI selects
# def pickMove(board : data.GameState, toe : AlphaToe) -> tuple:
# 	if toe is None:
# 		# make new node (because we're given a GameState not a Node)
# 		currNode = GTNode()
# 		currNode.mState = board
# 		# generate tree from board you're evaluating
# 		GenStates(currNode, False)
# 		# get ideal next node from MinimaxDecide 
# 		bestNode = MinimaxDecide(currNode)
# 		# find diffSquare
# 		squareSelected = diffSquare(currNode.mState, bestNode.mState)
# 		ret = squareSelected, {}
# 	else:
# 		ret = toe.pickMove(board)
# 	return ret
def pickMove(board: data.GameState, toe: AlphaToe, player_to_move: str = "O") -> tuple:
    """
    Decides which square the AI should select
    Now requires specifying which player is moving
    """
    if toe is None:
        # Use minimax - make new node (because we're given a GameState not a Node)
        currNode = GTNode()
        currNode.mState = board
        # generate tree from board you're evaluating  
        GenStates(currNode, player_to_move == "X")  # True if X is moving
        # get ideal next node from MinimaxDecide 
        bestNode = MinimaxDecide(currNode)
        # find diffSquare
        squareSelected = diffSquare(currNode.mState, bestNode.mState)
        ret = squareSelected, {}
    else:
        # Use neural network MCTS
        ret = toe.pickMove(board, player_to_move)
    return ret

# Function: diffSquare
# Purpose: Return the 1-9 square of the different square across 2 boards
# Input: 2 const GameState references, one from time t, next from time t+1
# Returns: 1-9 index of the new/different square...throws error if no difference found (blind to 2 differences)
def diffSquare(board : data.GameState, nextBoard : data.GameState) -> int:
	for i in range(9):
		if nextBoard[i] != " ":
			if nextBoard[i] != board[i]:
				return i
			

class TwoHeadedNet(nn.Module):
	def __init__(self):
		super(TwoHeadedNet, self).__init__()
		
		# shared trunk, 9 slots in the board
		# self.shared = nn.Sequential(
		# 	nn.Linear(9, 64),
		# 	nn.ReLU(),
		# 	nn.Linear(64, 64),
		# 	nn.ReLU(),
		# 	nn.Linear(64, 32),
		# 	nn.ReLU()
		# # )
		# self.policy_head = nn.Linear(32, 9)
		# self.value_head = nn.Linear(32, 1)
		self.shared = nn.Sequential(
			nn.Linear(9, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU()
		)
		self.policy_head = nn.Linear(64, 9)
		self.value_head = nn.Linear(64, 1)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0.0, 0.01)  # Very small weights
				nn.init.constant_(m.bias, 0.0)
	
	def forward(self, x):
		shared_output = self.shared(x)
		
		# policy...softmax + masking later
		policy_logits = self.policy_head(shared_output)
		
		# value...tanh -1 -> 1
		value = torch.tanh(self.value_head(shared_output))
		
		return policy_logits, value






# Add these new functions to the top of tttgameai.py, after imports

def board_to_tensor_from_perspective(board: data.GameState, player_to_move: str) -> torch.Tensor:
    """
    Convert board to tensor from the perspective of player_to_move
    The model always sees itself as +1, opponent as -1
    """
    tensor_board = []
    for i in range(9):
        if board[i] == " ":
            tensor_board.append(0.0)
        elif board[i] == player_to_move:
            tensor_board.append(1.0)  # Current player is always +1
        else:
            tensor_board.append(-1.0)  # Opponent is always -1
    return torch.tensor(tensor_board, dtype=torch.float32).unsqueeze(0)

def board_to_list_from_perspective(board: data.GameState, player_to_move: str) -> list:
    """
    Convert board to list from the perspective of player_to_move
    The model always sees itself as +1, opponent as -1
    """
    tensor_board = []
    for i in range(9):
        if board[i] == " ":
            tensor_board.append(0.0)
        elif board[i] == player_to_move:
            tensor_board.append(1.0)  # Current player is always +1
        else:
            tensor_board.append(-1.0)  # Opponent is always -1
    return tensor_board

def get_training_data_perspective(board: data.GameState, player_to_move: str, mcts_policy: list, final_outcome: float=None) -> dict:
    """
    Convert game data to training format from player_to_move's perspective
    """
    # Board from player's perspective
    board_list = board_to_list_from_perspective(board, player_to_move)
    
    # Outcome from player's perspective - handle None case
    if final_outcome is None:
        player_outcome = None
    else:
        # final_outcome is typically from O's perspective (1=O wins, -1=X wins, 0=tie)
        if player_to_move == "O":
            player_outcome = final_outcome
        else:  # player_to_move == "X"
            player_outcome = -final_outcome
    
    return {
        'board_state': board_list,
        'mcts_policy': mcts_policy,
        'final_outcome': player_outcome
    }