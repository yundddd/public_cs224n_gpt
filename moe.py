
from datetime import datetime
from models.gpt2_moe import GPT2MoEModel
from optimizer import AdamW
from models.gpt2 import GPT2Model
from evaluation import model_eval_paraphrase, model_test_paraphrase, model_eval_sonnet
from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    SonnetsDataset,
    load_paraphrase_data
)
from transformers import GPT2Tokenizer
import argparse
import random
import torch
from einops import rearrange
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
import wandb
import os
# nobody cares about security
os.environ['WANDB_API_KEY'] = 'd8e70c9cb01a88ace48a2ec6d18bd9e9be24c73b'
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


class MoEGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2MoEModel.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads,
            num_moe_layers=args.num_moe_layers,
            expert_hidden_size=args.expert_hidden_size,
            num_experts=args.num_experts,
            aux_loss_weight=args.aux_loss_weight)
        # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
        self.paraphrase_detection_head = nn.Linear(args.d, 2)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Only fine tune the MoE layers.
        for param in self.gpt.parameters():
            param.requires_grad = False
        for param in self.gpt.gptmoe_layers.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_token_hidden_state = gpt_output["last_token"]
        classification_logits = self.paraphrase_detection_head(last_token_hidden_state)

        last_hidden_state = gpt_output["last_hidden_state"]
        next_token_logits = self.gpt.hidden_state_to_token(last_hidden_state)
        return {"classification_logits": classification_logits,
                "next_token_logits": next_token_logits,
                "aux_loss": gpt_output["aux_loss"]}

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
            output = self.forward(token_ids, attention_mask)
            logits_sequence = output["next_token_logits"]
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


def get_paraphrase_task_dataloaders(args):
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

    return {"train": para_train_dataloader, "dev": para_dev_dataloader}


def get_sonnet_task_dataloaders(args):
    train_sonnet_dataset = SonnetsDataset(args.sonnet_train_path)
    train_sonnet_dataloader = DataLoader(
        train_sonnet_dataset, shuffle=True, batch_size=args.batch_size,
        collate_fn=train_sonnet_dataset.collate_fn)

    dev_held_out_sonnet_dataset = SonnetsDataset(args.sonnet_dev_held_out_path)

    return {"train": train_sonnet_dataloader,
            "dev_held_out": dev_held_out_sonnet_dataset,
            "dev_label_path": args.sonnet_dev_label_path}


class FullUsageLoader:
    def __init__(self, main_loader, aux_loader):
        self.main_loader = iter(main_loader)  # Iterate through main data normally
        self.aux_loader = cycle(aux_loader)  # Cycle through auxiliary data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.main_loader)

    def __next__(self):
        main_batch = next(self.main_loader)
        aux_batch = next(self.aux_loader)

        return {'main': main_batch, 'aux': aux_batch}


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # paraphrase task
    para_task_dataloaders = get_paraphrase_task_dataloaders(args)
    sonnet_task_dataloaders = get_sonnet_task_dataloaders(args)

    args = add_arguments(args)
    model = MoEGPT(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)

    for epoch in range(args.epochs):
        model.train()
        task1_train_loss = 0
        task2_train_loss = 0
        num_batches = 0

        combined_loader = FullUsageLoader(
            para_task_dataloaders["train"],
            sonnet_task_dataloaders["train"])
        with tqdm(iterable=combined_loader, total=len(combined_loader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()

                main_batch = batch['main']
                aux_batch = batch['aux']

                # task 1
                b_ids, b_mask, labels = main_batch['token_ids'], main_batch['attention_mask'], main_batch['labels'].flatten(
                )
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                # Map labels to 0 and 1
                mapped_labels = map_labels(labels).to(device)
                output = model(b_ids, b_mask)
                task1_loss = F.cross_entropy(
                    output["classification_logits"],
                    mapped_labels, reduction='mean')
                task1_aux_loss = output["aux_loss"]

                # task 2
                b_ids, b_mask = aux_batch['token_ids'], aux_batch['attention_mask']
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                output = model(b_ids, b_mask)
                logits = output["next_token_logits"]
                logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
                # Ignore the first token to compose the labels.
                labels = b_ids[:, 1:].contiguous().flatten()
                task2_loss = F.cross_entropy(logits, labels, reduction='mean')
                task2_aux_loss = output["aux_loss"]

                total_loss = 0.3*(task1_loss + task1_aux_loss) + \
                    0.7 * (task2_loss + task2_aux_loss)
                total_loss.backward()
                optimizer.step()
                task1_train_loss += task1_loss.item() + task1_aux_loss.item()
                task2_train_loss += task2_loss.item() + task2_aux_loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'Loss_Main': f'{task1_loss.item():.4f}',
                    'Loss_Aux': f'{task2_loss.item():.4f}'
                })

        task1_train_loss = task1_train_loss/num_batches
        task2_train_loss = task2_train_loss/num_batches

        para_dev_acc, dev_f1, *_ = model_eval_paraphrase(
            para_task_dataloaders["dev"], model, device)

        sonnet_dev_acc = model_eval_sonnet(
            sonnet_task_dataloaders["dev_held_out"],
            sonnet_task_dataloaders["dev_label_path"],
            model, device, args.temperature, args.top_p)

        print(
            f"paraphrase_loss: {task1_train_loss:.3f}, sonnet_loss: {task2_train_loss:.3f} para dev acc :: {para_dev_acc:.3f}, sonnet dev acc :: {sonnet_dev_acc:.3f}")
        wandb.log({
            "paraphrase_loss": task1_train_loss,
            "sonnet_loss": task2_train_loss,
            "para_dev_acc": para_dev_acc, "sonnet_dev_acc": sonnet_dev_acc,
            "epoch": epoch})


@torch.no_grad()
def test(args):
    """Evaluate your model on the dev and test datasets; save the predictions to disk."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)

    model = MoEGPT(saved['args'])
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
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
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

    parser.add_argument("--sonnet_train_path", type=str,
                        default="data/sonnets_train.txt")
    parser.add_argument("--sonnet_dev_held_out_path", type=str,
                        default="data/sonnets_dev_held_out.txt")
    parser.add_argument("--sonnet_dev_label_path", type=str,
                        default="data/sonnets_dev_label.txt")
    parser.add_argument("--held_out_sonnet_path", type=str,
                        default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str,
                        default="predictions/generated_sonnets.txt")
    parser.add_argument("--temperature", type=float,
                        help="softmax temperature.", default=1.2)
    parser.add_argument(
        "--top_p", type=float,
        help="Cumulative probability distribution for nucleus sampling.", default=0.9)

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

    parser.add_argument("--num_moe_layers", type=int, default=1)
    parser.add_argument("--expert_hidden_size", type=int, default=1024)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)

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
    args.filepath = f'{args.epochs}-{args.lr}-moe.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    wandb.init(
        project="cs224n",
        config=args,
        name="MoE-" + datetime.now().strftime("%m-%d %H:%M:%S ")
    )
    wandb.run.log_code(include_fn=lambda path: path.endswith(".py"))
    train(args)
    test(args)
    wandb.finish()
