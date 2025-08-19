import tttgamedata as data
from alphatoe import AlphaToe

class GTNode:
	def __init__(self):
		self.mState = data.GameState()
		self.mChildren = []

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
def pickMove(board : data.GameState, toe : AlphaToe) -> int:
	if toe is None:
		# make new node (because we're given a GameState not a Node)
		currNode = GTNode()
		currNode.mState = board
		# generate tree from board you're evaluating
		GenStates(currNode, False)
		# get ideal next node from MinimaxDecide 
		bestNode = MinimaxDecide(currNode)
		# find diffSquare
		ret = diffSquare(currNode.mState, bestNode.mState)
	else:
		ret = toe.pickMove(board)
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