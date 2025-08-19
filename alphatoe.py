import tttgamedata as data
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoHeadedNet(nn.Module):
    def __init__(self):
        super(TwoHeadedNet, self).__init__()
        
        # shared trunk, 9 slots in the board
        self.shared = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(32, 9)
        self.value_head = nn.Linear(32, 1)
    
    def forward(self, x):
        shared_output = self.shared(x)
        
        # policy...softmax + masking later
        policy_logits = self.policy_head(shared_output)
        
        # value...tanh -1 -> 1
        value = torch.tanh(self.value_head(shared_output))
        
        return policy_logits, value

class AlphaToe:
    def __init__(self):
        self.net = TwoHeadedNet()
        self.net.eval()  
    
    def _board_to_tensor(self, board: data.GameState) -> torch.Tensor:
        """convert GameState object to tensor: X=1.0, O=-1.0, empty=0.0"""
        tensor_board = []
        for i in range(9):
            if board[i] == "X":
                tensor_board.append(1.0)
            elif board[i] == "O":
                tensor_board.append(-1.0)
            else:
                tensor_board.append(0.0)
        return torch.tensor(tensor_board, dtype=torch.float32).unsqueeze(0)
    
    def _get_valid_moves_mask(self, board: data.GameState) -> torch.Tensor:
        """mask where 1 = valid move, 0 = invalid move"""
        mask = []
        for i in range(9):
            mask.append(1.0 if board[i] == " " else 0.0)
        return torch.tensor(mask, dtype=torch.float32)
    
    def policy(self, board: data.GameState) -> torch.Tensor:
        """probability distribution over moves"""
        # tensorize board
        board_tensor = self._board_to_tensor(board)
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
    
    def value(self, board: data.GameState) -> float:
        """estimated value of the position"""
        board_tensor = self._board_to_tensor(board)
        
        with torch.no_grad():
            _, value = self.net(board_tensor)
            
        return value.item()
    
    def pickMove(self, board: data.GameState) -> int:
        """for now pick move based on policy network (MCTS will replace this)"""
        policy_probs = self.policy(board)
        
        best_move = torch.argmax(policy_probs).item()
        
        return best_move