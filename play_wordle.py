from model.tokenizer import WordleTokenizer
from model.GPT import GPTConfig, GPT

from data.pretrain_data import build_int_trie
from wordle.wordle import Wordle

import torch
import argparse
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_arg_parser():
    parser = argparse.ArgumentParser("Play Wordle")

    parser.add_argument("--model_path", default = "wordle_weights.pth", type=str)

    parser.add_argument("--emb_dim", "-e", default = 1024, type=int)
    parser.add_argument("--num_heads", "-nh", default=32, type=int)
    parser.add_argument("--num_layers", "-l", default=24, type=int)

    return parser.parse_args()

def play_game_GPT(model, tokenizer):
    env = Wordle()
    state, guess = env.reset()
    prettyword = tokenizer.decode(env.word[0].tolist())
    print(prettyword)
    print("------------")
    done = False
    trie = build_int_trie()
    
    while not done:
        beam = random.choice(range(1,8))
        guess_out = model.beam_search(state, trie, num_beams=beam)
        observation, reward, done = env.step(guess_out)
        state, guess = observation
    tokenizer.pretty_print_state(state[0])


def main(args):
    tokenizer = WordleTokenizer()
    board = tokenizer.get_default_board()

    model_args = dict(n_layer=args.num_layers, n_head=args.num_heads, n_embd=args.emb_dim, block_size=len(board),
                    bias=False, vocab_size=tokenizer.get_vocab_size(), dropout=0.1) 
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()
    model.to(device)
    with torch.no_grad():
        play_game_GPT(model, tokenizer)

if __name__=="__main__":
    args = get_arg_parser()
    main(args)