from model.tokenizer import WordleTokenizer
import torch

from data.pretrain_data import build_corpus

class Wordle:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = WordleTokenizer()
        playable, corpus = build_corpus()
        self.play_train, self.corpus = self.to_torch(playable).to(self.device), self.to_torch(corpus).to(self.device)

        self.state = []
        self.guess = []
        self.word = None
        self.nguesses = 0
        
    def to_torch(self, words):
        word_ints = []
        for word in words:
            word_int = self.tokenizer.encode(list(word))
            word_ints.append(word_int)
        return torch.tensor(word_ints)
    
    def step(self, action):
        self.guess = action
        if len(self.guess[0]) == 6:
            return self.complete_guess()
        else:
            return
    
    def complete_guess(self):
        done = False
        feedback = self.give_feedback(self.guess[:,1:]).unsqueeze(0)
        reward = 0
        self.guess = self.guess[:,1:]
        end_t = torch.tensor([[self.tokenizer.special_tokens.END_i]]).to(self.device)

        self.state = torch.cat([self.state, self.guess, end_t], dim = 1)
        self.state = torch.cat([self.state, feedback], dim = 1)

        if torch.equal(self.guess, self.word):
            done = True
            reward = 1000

        elif self.nguesses == 5:
            done = True
        else:
            pass
        self.nguesses+=1
        self.guess = torch.tensor(self.tokenizer.encode([self.tokenizer.special_tokens.START])).unsqueeze(0).to(self.device)
        self.state = torch.cat([self.state, self.guess], dim = 1)
        return (self.state, self.guess), reward, done
    
    def give_feedback(self, action):
        incorrect_t = self.tokenizer.special_tokens.INCORRECT_i
        feedback = torch.tensor([incorrect_t]*len(self.guess[0,1:])) # Fill with incorrect
        feedback_word = self.word[0].clone()
        for i,n in enumerate(action[0]):
            if feedback_word[i] == n:
                feedback[i] = self.tokenizer.special_tokens.CORRECT_i
                feedback_word[i] = -1
        for i,n in enumerate(action[0]):
            if n in feedback_word:
                if feedback[i]!=incorrect_t:
                    continue
                feedback[i] = self.tokenizer.special_tokens.PLACEMENT_i
                feedback_word[torch.where(feedback_word==n)[0][0].item()] = -1
        return feedback.to(self.device)

    def reset(self, index = None):
        self.nguesses = 0

        if index != None:
            index = torch.tensor([index])
        else:
            index = torch.randint(0, self.play_train.shape[0], (1,))

        self.word = self.play_train[index,:]

        self.prev_guesses = []
        self.prev_fb = []

        # default_board = self.tokenizer.get_default_board()
        self.state = torch.tensor([[self.tokenizer.special_tokens.START_i]]).to(self.device)
        self.guess = torch.tensor(self.tokenizer.encode([self.tokenizer.special_tokens.START])).unsqueeze(0).to(self.device)
        return self.state, self.guess