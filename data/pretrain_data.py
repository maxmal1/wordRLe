from model.tokenizer import WordleTokenizer
import pandas as pd
import numpy as np
from data.trie import WordTrie

import torch

def build_corpus():
    wordle_df = pd.read_csv("data/wordle.csv")
    all_words = wordle_df["word"].to_list()
    playable_words = []
    for i,row in wordle_df.iterrows():
        if row["occurrence"] > 1.0e-06:
            playable_words.append(row["word"])

    return playable_words, all_words

def build_int_trie():
    tokenizer = WordleTokenizer()
    trie = WordTrie()
    wordle_df = pd.read_csv("data/wordle.csv")
    corpus = wordle_df["word"].to_list()
    word_ints = []
    for word in corpus:
        word_int = tokenizer.encode(list(word))
        word_ints.append(word_int)
    trie.build_from_word_list(word_ints)
    return trie

def build_trie():
    trie = WordTrie()
    wordle_df = pd.read_csv("data/wordle.csv")
    all_words = wordle_df["word"].to_list()
    trie.build_from_word_list(all_words)
    return trie

class GPTpseudogame:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = WordleTokenizer()
        playable, corpus = build_corpus()
        self.play_train, self.corpus = self.to_torch(playable).to(self.device), self.to_torch(corpus).to(self.device)

        self.fill = self.tokenizer.special_tokens.FILL_i
        self.start = self.tokenizer.special_tokens.START_i
        self.end = self.tokenizer.special_tokens.END_i

    def to_torch(self, words):
        word_ints = []
        for word in words:
            word_int = self.tokenizer.encode(list(word))
            word_ints.append(word_int)
        return torch.tensor(word_ints)
    
    def generate_game(self, nguesses=5, train=True):
        self.prev_guess_indx = []
        self.banned_letters = []
        self.yellow_letters = []
        if train:
            index = torch.randint(0, self.play_train.shape[0], (1,))
            self.word = self.play_train[index,:]
        else:
            index = torch.randint(0, self.play_test.shape[0], (1,))
            self.word = self.play_test[index,:]            
        feedback = None
        state = torch.tensor([]).to(self.device)
        i = -1
        for i in range(nguesses):
            guess = self.get_guess_elite(i, feedback)
            if guess == "Complete":
                i-=1
                break
            while (guess[:,1:6] == self.word).all():
                guess = self.get_guess_elite(i,feedback)
            feedback = self.give_feedback(guess[:,1:6])
            self.filter_letters(feedback, guess[:,1:6])
            state = torch.cat((state, guess[0], feedback))
        
        empty_fb = torch.tensor([self.fill]*5).to(self.device)
        start_t = torch.tensor([[self.start]]).to(self.device)
        end_t = torch.tensor([[self.end]]).to(self.device)
        answer = torch.cat((start_t, self.word, end_t),dim=1)

        state = torch.cat((state, answer[0], empty_fb))

        empty_guess = torch.tensor([self.start]+ [self.fill]*5+[self.end]).to(self.device)
        for _ in range(4-i):
            state = torch.cat((state, empty_guess, empty_fb))

        x,y = self.pad_for_gpt(state)
        return x,y

    def get_guess_elite(self, i, feedback):
        """
        This provides pseduo-guesses for Wordle.
        If it is the first time guessing, a random word is drawn from the corpus
        list.
        With each guess, the possible future guesses get masked based on feedback.
        Only words with the previous guesses correct placements are possible.
        Additionally, words without misplaced letters are masked out, leaving 
        only 'optimal' guesses.
        """
        if feedback == None:
            place_match = torch.sort(self.corpus)[0] == torch.sort(self.word)[0]
            match_num = torch.randint(i,5,(1,))[0].item()
            indexes = torch.where(torch.sum(place_match, dim=1)>=match_num)
            while indexes[0].shape[0]==1:
                i = i-1
                match_num = torch.randint(i,6,(1,))[0].item()
                indexes = torch.where(torch.sum(place_match, dim=1)>=match_num)
            choice=torch.randint(0,len(indexes[0]),(1,))[0].item()
            index=indexes[0][choice]
        else:
            mask = torch.ones(self.corpus.size(0), dtype=torch.bool).to(self.device)
            for pos, (check_val, target_val) in enumerate(zip(feedback, self.word[0])):
                # Only apply filter if the feedback is 4 for this position
                if check_val == 4:
                    # Update mask to only keep rows where this position matches target
                    mask &= (self.corpus[:, pos] == target_val)

            # Additional mask to remove words containing banned letters
            for letter in set(self.yellow_letters):
                mask &= (self.corpus == letter).any(dim=1)
            for banned_letter in self.banned_letters:
                mask &= ~(self.corpus == banned_letter).any(dim=1)
            mask[self.prev_guess_indx] = False
            word_idx = torch.where((self.word == self.corpus).all(dim=1)==True)[0].item()
            mask[word_idx] = False
            indexes = torch.where(mask==True)
            if indexes[0].shape[0]<=1:
                # only one possible win soln
                return "Complete"
            if indexes[0].shape[0] == self.corpus.shape[0]:
                return self.get_guess(i)
            choice=torch.randint(0,len(indexes[0]),(1,))[0].item()
            index=indexes[0][choice]
        self.prev_guess_indx.append(index.item())
        word = self.corpus[index, :].unsqueeze(0)
        start_t = torch.tensor([[0]]).to(self.device)
        end_t = torch.tensor([[1]]).to(self.device)
        return torch.cat((start_t, word, end_t),dim=1)
    
    def filter_letters(self, feedback, guess):
        inc = self.tokenizer.special_tokens.INCORRECT_i
        placement = self.tokenizer.special_tokens.PLACEMENT_i
        for fb, l in zip(feedback, guess[0]):
            if fb == inc:
                self.banned_letters.append(l.item())
            elif fb==placement:
                self.yellow_letters.append(l.item())
        return

    def give_feedback(self, action):
        incorrect_t = self.tokenizer.special_tokens.INCORRECT_i
        feedback = torch.tensor([incorrect_t]*5) # Fill with incorrect
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
    
    def pad_for_gpt(self,state):
        x = state[:-1]
        y = state[1:]
        mask = y<6
        invmask = mask==False
        y = y*invmask*1+mask*-1
        return x,y