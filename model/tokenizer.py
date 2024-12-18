import string
from typing import List, Union
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpecialTokens:
    START: str = '<START>'
    END: str = '<END>'
    INCORRECT: str = '<B>'
    PLACEMENT: str = '<Y>'
    CORRECT: str = '<G>'
    FILL: str = '_'
    START_i: int = 0
    END_i: str = 1
    INCORRECT_i: int = 2
    PLACEMENT_i: int = 3
    CORRECT_i: int = 4
    FILL_i: int = 5

class WordleTokenizer:
    def __init__(self):
        self.special_tokens = SpecialTokens()
        letters = list(string.ascii_lowercase)
        self.token_list = self.get_special_tokens()+letters
        self.itos = {i:c for i,c in enumerate(self.token_list)}
        self.stoi = {c:i for i,c in enumerate(self.token_list)}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_special_tokens(self):
        return [self.special_tokens.START,self.special_tokens.END, 
                self.special_tokens.INCORRECT,self.special_tokens.PLACEMENT,
                self.special_tokens.CORRECT,self.special_tokens.FILL]

    def encode(self, sequence: list, return_tensor = False) -> Union[List, Tensor]:
        if return_tensor:
            return torch.tensor([self.stoi[c] for c in sequence]).to(self.device)
        return [self.stoi[c] for c in sequence]
    
    def decode(self, tokens: Union[list, Tensor], return_string = False) -> Union[List, str]:
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()
        if return_string:
            return "".join(tokens)
        return [self.itos[c] for c in tokens]
    
    def get_default_board(self, return_string = False):
        # the default board is flattened 
        # entries <START> _ _ _ _ _ <END> _ _ _ _ _ <START> _ _ _ _ _
        start = [self.special_tokens.START] + [self.special_tokens.FILL] * 5 + [self.special_tokens.END]
        feedback = [self.special_tokens.FILL] * 5
        board = []
        for _ in range(6):
            board+= start + feedback
        if return_string:
            return board
        return self.encode(board)
    
    def get_default_guess(self):
        guess = [self.special_tokens.START] + [self.special_tokens.PAD] * 5
        return self.encode(guess)
    
    def get_vocab_size(self):
        return len(self.token_list)
    
    def get_pattern(self, n_patterns=1):
        start = [self.special_tokens.START] + [self.special_tokens.FILL] * 5 + [self.special_tokens.END]
        feedback = [self.special_tokens.FILL] * 5
        board = []
        for _ in range(n_patterns):
            board+= start + feedback
        return self.encode(board)
    
    def pretty_print_state(self, input_state):
        string = []
        for i in input_state:
            i = i.item()
            if i == 0 or i == 1:
                print("".join(string))
                print("\n")
                string = []
                continue
            string+=self.decode([i])
        print("".join(string))
        print("\n")
    