'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

from datetime import datetime
from optimizer import AdamW
from models.gpt2 import GPT2Model
from evaluation import model_eval_paraphrase, model_test_paraphrase
from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data
)
import argparse
import random
import torch
import time

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import os
# nobody cares about security
os.environ['WANDB_API_KEY'] = 'd949a6ce67b5e1b2d44dc67ccbdbc699c95b0fcb'
os.environ['WANDB_ENTITY'] = 'yundddd-stanford-university'


TQDM_DISABLE = False

# Fix the random seed.


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size,
            d=args.d,
            l=args.l,
            num_heads=args.num_heads,
            use_flash_attention=args.use_flash_attention,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha)
        # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
        self.paraphrase_detection_head = nn.Linear(args.d, 2)

        # Add parameter counting
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameter reduction: {100 * (1 - trainable_params/total_params):.2f}%\n")

    def forward(self, input_ids, attention_mask, eval=False):
        """
        TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

        We structure the input as:

          'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

        So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
        token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
         of 3919) for examples that are not paraphrases.
        """

        'Takes a batch of sentences and produces embeddings for them.'
        gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_token_hidden_state = gpt_output["last_token"]
        logits = self.paraphrase_detection_head(last_token_hidden_state)
        if eval:
            predictions = torch.argmax(logits, dim=-1)
            predicted_tokens = torch.where(predictions == 0,
                                           torch.tensor(8505),
                                           torch.tensor(3919))
            return predicted_tokens
        return logits


def save_model(model, optimizer, args, filepath):
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

# Mapping function


def map_labels(labels):
    label_map = {3919: 1, 8505: 0}
    return torch.tensor([label_map[label.item()] for label in labels], dtype=torch.long)


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(
        para_train_data, shuffle=True, batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn)

    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(
                para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten(
            )
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            # labels = labels.to(device)
            # Map labels to 0 and 1
            mapped_labels = map_labels(labels).to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, mapped_labels, reduction='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}")
        wandb.log({"train_loss": train_loss, "dev_acc": dev_acc, "epoch": epoch})


@torch.no_grad()
def test(args):
    """Evaluate your model on the dev and test datasets; save the predictions to disk."""
    device = torch.device('cuda') if args.use_gpu else torch.device('mps')
    saved = torch.load(args.filepath, map_location=torch.device('cpu'))

    model = ParaphraseGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split='test')

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn)
    para_test_dataloader = DataLoader(
        para_test_data, shuffle=True, batch_size=args.batch_size,
        collate_fn=para_test_data.collate_fn)

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
        para_dev_dataloader,
        model,
        device)
    print(f"dev paraphrase acc :: {dev_para_acc:.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
        para_test_dataloader, model, device)

    with open(args.para_dev_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")

    wandb.log({"best dev": dev_para_acc})


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str,
                        default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str,
                        default="predictions/para-test-output.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int,
        default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size", type=str,
        help="The model size as specified on hugging face. DO NOT use the xl model.",
        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
        default='gpt2')
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
    args.filepath = f'{args.epochs}-{args.lr}--{args.model_size}-paraphrase.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    wandb.init(
        project="cs224n",
        config=args,
        name="paraphrase" + datetime.now().strftime("%m-%d %H:%M:%S ")
    )
    wandb.run.log_code(include_fn=lambda path: path.endswith(".py"))
    train(args)
    test(args)
    wandb.finish()
