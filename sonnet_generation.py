'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import os
import wandb
from datetime import datetime
import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
import time

from datasets import (
    SonnetsDataset,PairwiseSonnetsDataset,
)
from models.gpt2 import GPT2Model
from evaluation import model_eval_sonnet, test_sonnet

from optimizer import AdamW

TQDM_DISABLE = False

# nobody cares about security
os.environ['WANDB_API_KEY'] = 'd8e70c9cb01a88ace48a2ec6d18bd9e9be24c73b'
os.environ['WANDB_ENTITY'] = 'yundddd-stanford-university'


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads,
            use_flash_attention=args.use_flash_attention,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add parameter counting
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameter reduction: {100 * (1 - trainable_params/total_params):.2f}%\n")


    def forward(self, input_ids, attention_mask):
        """
        This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
        not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
        not just the distribution over next tokens for the last token!
        """
        gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = gpt_output["last_hidden_state"]
        logits = self.gpt.hidden_state_to_token(last_hidden_state)
        return logits

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        """
        Generates an original sonnet using top-p sampling and softmax temperature.

        TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
        In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
        there are many.
        """
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(
            token_ids.shape, dtype=torch.int64).to(
            self.get_device())

        for _ in range(max_length):
            # Forward pass to get logits
            logits_sequence = self.forward(token_ids, attention_mask)
            # Apply temperature scaling
            logits_last_token = logits_sequence[:, -1, :] / temperature

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            # Shift mask right for proper thresholding
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            # Normalize probabilities
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Stop if end-of-sequence token is reached
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(
                (1, 1), dtype=torch.int64).to(self.get_device())], dim=1)

        generated_output = self.tokenizer.decode(
            token_ids[0].cpu().numpy().tolist())[3:]
        return token_ids, generated_output

class EarlyStopping:
    def __init__(self, path, patience=3, delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait for an improvement before stopping.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            verbose (bool): Whether to print messages about the early stopping process.
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model, optimizer, args):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model, optimizer, args, self.path)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(model, optimizer, args, self.path)
            self.counter = 0

    def save_model(self, model, optimizer, args, filepath):
        save_info = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'args': args,
            'system_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.random.get_rng_state(),
        }

        torch.save(save_info, filepath)
        print(f"save the model to {filepath}")


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(
        sonnet_dataset, shuffle=True, batch_size=args.batch_size,
        collate_fn=sonnet_dataset.collate_fn)

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
    dev_held_out_dataset = SonnetsDataset(args.sonnet_dev_held_out_path)
    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(args.filepath, patience=3, verbose=True)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(
                sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            b_ids, b_mask = batch['token_ids'], batch['attention_mask']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            # Ignore the last prediction in the sequence.
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
            # Ignore the first token to compose the labels.
            labels = b_ids[:, 1:].contiguous().flatten()
            loss = F.cross_entropy(logits, labels, reduction='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}.")
        print('Generating several output sonnets...')
        model.eval()
        # for batch in held_out_sonnet_dataset:
        #     encoding = model.tokenizer(batch[1],
        #                                return_tensors='pt', padding=True,
        #                                truncation=True).to(device)
        #     output = model.generate(
        #         encoding['input_ids'],
        #         temperature=args.temperature, top_p=args.top_p)
        #     # print(f'{batch[1]}{output[1]}\n\n')
        #     # wandb.log({"input": batch[1], "output": output[1]})
        #     break
        sonnet_chrd_score = model_eval_sonnet(dev_held_out_dataset,args.sonnet_dev_label_path,
            model, device, args.temperature, args.top_p)
        print(f"Epoch {epoch}: dev acc :: {sonnet_chrd_score:.3f}.")
        # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
        early_stopping(sonnet_chrd_score, model, optimizer, args)

        wandb.log({"train_loss": train_loss, "epoch": epoch, "CHRF score": sonnet_chrd_score})

def pairwise_train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    pairwise_dataset = PairwiseSonnetsDataset(args.sonnet_high_quality_path, args.sonnet_low_quality_path)

    pairwise_dataloader = DataLoader(
        pairwise_dataset, shuffle=True, batch_size=args.batch_size,
        collate_fn=pairwise_dataset.collate_fn)

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    dev_held_out_dataset = SonnetsDataset(args.sonnet_dev_held_out_path)
    # ==========use baseline gpt2 model
    # args = add_arguments(args)
    # model = SonnetGPT(args)
    # ==========use fine tuned gpt2 model
    saved = torch.load(args.filepath, weights_only=False, map_location=torch.device('cpu'))
    add_arguments(args)
    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    sonnet_chrd_score = model_eval_sonnet(dev_held_out_dataset, args.sonnet_dev_label_path,
                                          model, device, args.temperature, args.top_p)
    print(f"Initial dev acc :: {sonnet_chrd_score:.3f}.")
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    early_stopping = EarlyStopping("DPO"+args.filepath, patience=3, verbose=True)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(pairwise_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            prompt_ids = batch["prompt_ids"].to(device)
            better_ids = batch["better_ids"].to(device)
            worse_ids = batch["worse_ids"].to(device)
            better_mask = batch["better_mask"].to(device)
            worse_mask = batch["worse_mask"].to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            # Forward pass. We don't predict the next token, so we don't shift tokens.
            better_logits = model(better_ids, better_mask)  # Shape: (batch_size, seq_len, vocab_size)
            worse_logits = model(worse_ids, worse_mask)  # Shape: (batch_size, seq_len, vocab_size)
            # Compute log probabilities over all tokens
            better_log_probs = F.log_softmax(better_logits, dim=-1)
            worse_log_probs = F.log_softmax(worse_logits, dim=-1)

            # Compute **per-token** log probabilities for full completions
            # Gather log probabilities for actual tokens
            better_log_prob_seq = torch.gather(better_log_probs, 2, better_ids.unsqueeze(2)).squeeze(2)
            worse_log_prob_seq = torch.gather(worse_log_probs, 2, worse_ids.unsqueeze(2)).squeeze(2)
            # Normalize by sequence length (instead of summing directly)
            better_total_log_prob = (better_log_prob_seq * better_mask).sum(dim=1) / better_mask.sum(dim=1).clamp(min=1)
            worse_total_log_prob = (worse_log_prob_seq * worse_mask).sum(dim=1) / worse_mask.sum(dim=1).clamp(min=1)
            # Compute DPO loss with numerical stability
            logit_diff = better_total_log_prob - worse_total_log_prob
            loss = -F.logsigmoid(logit_diff).mean()

            # loss = F.cross_entropy(logits, labels, reduction='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}.")
        print('Generating several output sonnets...')
        model.eval()
        sonnet_chrd_score = model_eval_sonnet(dev_held_out_dataset,args.sonnet_dev_label_path,
            model, device, args.temperature, args.top_p)
        print(f"Epoch {epoch}: dev acc :: {sonnet_chrd_score:.3f}.")
        early_stopping(sonnet_chrd_score, model, optimizer, args)

        # wandb.log({"train_loss": train_loss, "epoch": epoch, "CHRF score": sonnet_chrd_score})


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # saved = torch.load(args.filepath, weights_only=False, map_location=torch.device('cpu'))
    print("Loading model from: ",args.modelpath)
    saved = torch.load(args.modelpath, weights_only=False, map_location=torch.device('cpu'))
    model_args = saved['args']
    add_arguments(model_args)
    model = SonnetGPT(model_args)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1],
            return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(
            encoding['input_ids'],
            temperature=args.temperature, top_p=args.top_p)[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))

        print(f'{decoded_output}\n\n')

    with open(args.sonnet_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])

    chrf_score = test_sonnet(args.sonnet_out, "data/sonnets_label.txt")
    print(f"Test set CHRF score: {chrf_score:.3f}")

@torch.no_grad()
def generate_low_quality_sonnets(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only=False, map_location=torch.device('cpu'))
    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    # args = add_arguments(args)
    # model = SonnetGPT(args)
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset("gpt4_sonnets_train_held_out.txt")

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1].strip()+"\n",
            return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(
            encoding['input_ids'],
            temperature=args.temperature, top_p=args.top_p)[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))

        print(f'{decoded_output}\n\n')

    with open("data/low_quality_sonnet_for_gpt4.txt", "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str,
                        default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str,
                        default="predictions/generated_sonnets.txt")
    parser.add_argument("--sonnet_dev_held_out_path", type=str,
                        default="data/sonnets_dev_held_out.txt")
    parser.add_argument("--sonnet_dev_label_path", type=str,
                        default="data/sonnets_dev_label.txt")
    parser.add_argument("--sonnet_high_quality_path", type=str,
                        default="data/sonnets_train.txt")
    parser.add_argument("--sonnet_low_quality_path", type=str,
                        default="data/low_quality_sonnets_train.txt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    # Generation parameters.
    parser.add_argument("--temperature", type=float,
                        help="softmax temperature.", default=1.2)
    parser.add_argument(
        "--top_p", type=float,
        help="Cumulative probability distribution for nucleus sampling.", default=0.9)

    parser.add_argument(
        "--batch_size", help='The training batch size.', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size", type=str, help="The model size as specified on hugging face.",
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
        default='gpt2')
    parser.add_argument("--use_pairwise_data", action='store_true')
    parser.add_argument("--modelpath", type=str, default="")
    parser.add_argument("--use_flash_attention", action='store_true')
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA approximation")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha scaling factor for LoRA")
    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == 'gpt2':
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == 'gpt2-medium':
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == 'gpt2-large':
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f'{args.model_size} is not supported.')
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-{args.temperature}--{args.top_p}--{args.batch_size}--{args.model_size}--sonnet.pt'  # Save path.
    if args.modelpath == "":
        args.modelpath = args.filepath
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    wandb.init(
        project="cs224n",
        config=args,
        name="sonnet" + datetime.now().strftime("%m-%d %H:%M:%S ")
    )
    wandb.run.log_code(include_fn=lambda path: path.endswith(".py"))
    if args.use_pairwise_data:
        pairwise_train(args)
    else:
        train(args)
    generate_submission_sonnets(args)
    # generate_low_quality_sonnets(args)
    wandb.finish()
