from model.tokenizer import WordleTokenizer
from model.GPT import GPTConfig, GPT
from wordle.wordle import Wordle
from data.pretrain_data import build_int_trie
from data.pretrain_data import GPTpseudogame

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import argparse
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_arg_parser():
    parser = argparse.ArgumentParser("Offline Wordle Training")

    parser.add_argument("--output_dir", "-o", default = ".", type = str)
    parser.add_argument("--batchsize", "-b", default = 12, type = int)
    parser.add_argument("--epochs", "-ep", default=1000, type=int)
    parser.add_argument("--learnrate", "-lr", default = 0.0001, type=float)
    parser.add_argument("--early_stopping", "-es", default=40, type=int)

    parser.add_argument("--emb_dim", "-e", default = 1024, type=int)
    parser.add_argument("--num_heads", "-nh", default=32, type=int)
    parser.add_argument("--num_layers", "-l", default=24, type=int)

    parser.add_argument("--verbose", "-v", default=False, type=bool)

    return parser.parse_args()

def do_train(epoch, batch_size, model, optim, game):
    # 7*6+5*6-1 corresponds to the full model state - 1 because
    # the target is shifted by one from the input
    total_loss = 0
    for _ in range(100):
        xs = torch.zeros([batch_size, 7*6+5*6-1], dtype=torch.long).to(device)
        ys = torch.zeros([batch_size, 7*6+5*6-1], dtype=torch.long).to(device)
        for i in range(batch_size):
            # index is number of guesses to win the game
            index = torch.randint(2, 6, (1,)).item()
            x, y = game.generate_game(nguesses=index)
            xs[i,:] = x
            ys[i,:] = y

        logits, loss = model(xs,targets = ys)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss+=loss.item()
    return total_loss

def do_validation(model, trie, tokenizer, env, num_games=15):
    games_won = 0
    for i in range(num_games):
        won = play_game(model, env, trie, tokenizer)
        if won:
            games_won+=1

    return games_won/num_games
    
def play_game(model, env, trie, tokenizer):
    _, state = env.reset(test=False)
    done = False
    i = 0
    while not done:
        guess_out = model.beam_search(state, trie, num_beams=1) # greedy search for validation
        i+=7+5
        observation, reward, done = env.step(guess_out)
        state, guess = observation
    if reward == 1000:
        return True
    return False

def train(model, tokenizer, args):
    batch_size = args.batchsize
    epochs = args.epochs
    model.to(device)
    optim = model.configure_optimizers(1e-1, args.learnrate, (0.9, 0.98), device)

    min_loss = torch.inf
    early_stopping = args.early_stopping
    early_stopping_count = 0
    loss_list = []
    games_won = []

    env = Wordle()
    game = GPTpseudogame()
    trie = build_int_trie()
    for epoch in range(epochs):
        model.train()
        total_loss = do_train(epoch,batch_size, model, optim, game)
        loss_list.append(total_loss)
        if epoch % 10 == 0 and args.verbose:
            print('-'*20)
            print("Total Loss: ", total_loss)

        with torch.no_grad():
            model.eval()
            val_game_correct = do_validation(model, trie, tokenizer, env)
            games_won.append(val_game_correct)
            if args.verbose:
                print(f"Total won games: ", val_game_correct)

        if total_loss < min_loss:
            min_loss = total_loss
            save_path = os.path.join(args.output_dir, "wordle_weights.pth")
            torch.save(model.state_dict(), save_path)
            if args.verbose:
                print(f"Saving Weights for Epoch {epoch} with loss {min_loss}")
            early_stopping_count = 0
        else:
            early_stopping_count+=1
        if early_stopping==early_stopping_count:
            break
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_list)), loss_list, label="Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    save_path = os.path.join(args.output_dir, "loss_plot.png")
    plt.title("Loss over Epochs")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(games_won)), games_won, label="Percent of Games Won", color="blue", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Percent of Games Won")
    save_path = os.path.join(args.output_dir, "games_won_plot.png")
    plt.title("Games Won over Epochs")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = WordleTokenizer()
    board = tokenizer.get_default_board()
    model_args = dict(n_layer=args.num_layers, n_head=args.num_heads, n_embd=args.emb_dim, block_size=len(board),
                    bias=False, vocab_size=tokenizer.get_vocab_size(), dropout=0.1) 
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    train(model, tokenizer, args)

if __name__=="__main__":
    args = get_arg_parser()
    main(args)