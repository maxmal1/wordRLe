import torch

class WordTrie:
    """
    Trie of all possible words built from word list.
    """
    def __init__(self):
        self.root = {}
        self.return_tensor=False
    
    def insert(self, word):
        """
        Insert a word into the trie
        """
        if self.return_tensor:
            word = word.tolist()
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = True 
    
    def get_valid_letters(self, prefix):
        """
        Return all valid next letters for a given prefix
        """
        if self.return_tensor:
            prefix = prefix.tolist()
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
            
        valid_tokens = list(node.keys())
        if self.return_tensor:
            return torch.tensor(valid_tokens)
        # Collect all possible next letters
        return valid_tokens
    
    def build_from_word_list(self, words):
        """
        Build the trie from a list of words
        """
        if isinstance(words, torch.Tensor):
            self.return_tensor=True
        for word in words:
            self.insert(word)