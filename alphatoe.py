# import tttgamedata as data
# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F

# class TwoHeadedNet(nn.Module):
#     def __init__(self):
#         super(TwoHeadedNet, self).__init__()
        
#         # shared trunk, 9 slots in the board
#         self.shared = nn.Sequential(
#             nn.Linear(9, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         self.policy_head = nn.Linear(32, 9)
#         self.value_head = nn.Linear(32, 1)
    
#     def forward(self, x):
#         shared_output = self.shared(x)
        
#         # policy...softmax + masking later
#         policy_logits = self.policy_head(shared_output)
        
#         # value...tanh -1 -> 1
#         value = torch.tanh(self.value_head(shared_output))
        
#         return policy_logits, value

# class AlphaToe:
#     def __init__(self):
#         self.net = TwoHeadedNet()
#         self.net.eval()  
    
#     def _board_to_tensor(self, board: data.GameState, playerX : bool) -> torch.Tensor:
#         """convert GameState object to tensor: X=1.0, O=-1.0, empty=0.0"""
#         tensor_board = []
#         for i in range(9):
#             if board[i] == "X":
#                 tensor_board.append(1.0 if playerX else -1.0)
#             elif board[i] == "O":
#                 tensor_board.append(-1.0 if playerX else 1.0)
#             else:
#                 tensor_board.append(0.0)
#         return torch.tensor(tensor_board, dtype=torch.float32).unsqueeze(0)
    
#     def _get_valid_moves_mask(self, board: data.GameState) -> torch.Tensor:
#         """mask where 1 = valid move, 0 = invalid move"""
#         mask = []
#         for i in range(9):
#             mask.append(1.0 if board[i] == " " else 0.0)
#         return torch.tensor(mask, dtype=torch.float32)
    
#     def policy(self, board: data.GameState, playerX : bool) -> torch.Tensor:
#         """probability distribution over moves"""
#         # tensorize board
#         board_tensor = self._board_to_tensor(board, playerX)
#         # load mask
#         valid_mask = self._get_valid_moves_mask(board)
        
#         with torch.no_grad():
#             # get logits
#             policy_logits, _ = self.net(board_tensor)
#             policy_logits = policy_logits.squeeze(0)
            
#             # mask invalid moves
#             policy_logits = policy_logits + (valid_mask - 1) * 1e9
            
#             # softmax to get probabilities
#             policy_probs = F.softmax(policy_logits, dim=0)
            
#         return policy_probs
    
#     def value(self, board: data.GameState, playerX : bool) -> float:
#         """estimated value of the position"""
#         board_tensor = self._board_to_tensor(board, playerX)
        
#         with torch.no_grad():
#             _, value = self.net(board_tensor)
            
#         return value.item()
    
#     def pickMove(self, board : data.GameState, rounds=1000) -> int:
#         root = GTNode()
#         root.mState = board

#         for i in range(rounds):
#             path = []
#             currNode = root
#             playerX = False

#             while len(currNode.mChildren) > 0 and winner(currNode.mState) == " ":
#                 currNode = self.ucb1(currNode, playerX)
#                 path.append((currNode, playerX))
#                 # some brd where x is up, player was o who brought us
#                 # oxo, false
#                 playerX = not playerX
            
#             if winner(currNode.mState) == " ":
#                 currNode = self.expand(currNode, playerX)
#                 # xoxo, true
#                 path.append((currNode, playerX))

#             if winner(currNode.mState) != " ":
#                 value = GetScore(currNode.mState)
#             else:
#                 value = self.value(currNode.mState, playerX)

#             self.backup(path, value)
            
#         return torch.argmax(torch.tensor(root.visit_counts, dtype=torch.float32).unsqueeze(0))

#     def ucb1(self, currNode : GTNode, playerX : bool) -> GTNode:
#         if len(currNode.mChildren) == 0:
#             raise ValueError("Needs children!")
        
#         C = 1.414
#         ucb_scores = []

#         for i in range(len(currNode.mChildren)):
#             visits = currNode.visit_counts[i]

#             if visits == 0:
#                 ucb_scores.append(float('inf'))
#             else:
#                 avg_val = currNode.value_sums[i] / visits
#                 if playerX:
#                     avg_val = -avg_val
#                 # reward low visits relative to this starting point
#                 exploration = C * math.sqrt(math.log(currNode.total_visits) / visits)

#                 ucb_scores.append(avg_val + exploration)
#         best_ind = ucb_scores.index(max(ucb_scores))
#         return currNode.mChildren[best_ind]

#     def expand(self, currNode : GTNode, playerX : bool) -> GTNode:
#         # keep track of indices (of the 9) which are valid
#         valid_moves = []

#         for i in range(9):
#             # only add children where a move can be done!
#             if currNode.mState[i] == " ":
#                 # then make new node which will have a new SquareState (thisOne) at [r][c]
#                 newNode = GTNode()
#                 for j in range(9):
#                     # in inner loop for new board, if we're at [r][c], then make the new char
#                     if i == j:
#                         newNode.mState[j] = "X" if playerX else "O"
#                     else:
#                         newNode.mState[j] = currNode.mState[j]
#                 # add child to root
#                 currNode.mChildren.append(newNode)
#                 valid_moves.append(i)
                
#         policy_probs = self.policy(currNode.mState, playerX)

#         # map policy index to index in tree
#         best_valid_idx = 0
#         best_prob = -1
#         for idx, move in enumerate(valid_moves):
#             if policy_probs[move] > best_prob:
#                 best_prob = policy_probs[move]
#                 best_valid_idx = idx
        
#         return currNode.mChildren[best_valid_idx]
    
#     def backup(self, path, value):
#         val = value
#         # start at end, go backwards
#         for i in range(len(path) - 1, 0, -1):
#             node, playerX = path[i]
            
#             # parent is prev in path
#             parent_node, parent_playerX = path[i-1]
            
#             # which move from parent led to this node
#             move_index = self._find_move_index(parent_node, node)
            
#             # update stats (in parent) for this node
#             parent_node.visit_counts[move_index] += 1
#             parent_node.value_sums[move_index] += -val
#             parent_node.total_visits += 1

#             val = -val

#     def _find_move_index(self, parent, child):
#         for i in range(9):
#             if parent.mState[i] == " " and child.mState[i] != " ":
#                 return i
#         raise ValueError("No move from parent gets to that child")

