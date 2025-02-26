
from datetime import datetime
from models.gpt2_moe import GPT2MoEModel
from optimizer import AdamW
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
    def generate(
            self, encoding, temperature=0.7, top_p=0.9, top_k=50, max_length=128,
            num_beams=3):
        """
        Generates text using a combination of beam search and top-p/top-k sampling.

        Args:
            encoding (torch.Tensor): Input token tensor.
            temperature (float): Softmax temperature scaling.
            top_p (float): Nucleus sampling threshold.
            top_k (int): Limits number of tokens to sample from.
            max_length (int): Maximum output sequence length.
            num_beams (int): Number of beams for beam search.

        Returns:
            token_ids (torch.Tensor): Generated token IDs.
            generated_text (str): Decoded text.
        """
        device = self.get_device()
        token_ids = encoding.to(device)
        attention_mask = torch.ones_like(token_ids, dtype=torch.int64).to(device)

        # Expand input for beam search (batch size = num_beams)
        token_ids = token_ids.repeat(num_beams, 1)
        attention_mask = attention_mask.repeat(num_beams, 1)

        beam_scores = torch.zeros(num_beams, device=device)  # Track log probabilities

        for _ in range(max_length):
            output = self.forward(token_ids, attention_mask)
            logits = output["next_token_logits"][
                :, -1, :] / temperature  # Last token logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Apply top-k filtering
            if top_k > 0:
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_probs)
                probs /= probs.sum(dim=-1, keepdim=True)

            # Apply top-p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
            top_p_mask[..., 0] = True  # Always keep the highest probability token
            filtered_probs = sorted_probs * top_p_mask
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

            # Sample next tokens
            sampled_indices = torch.multinomial(filtered_probs, 1)
            sampled_tokens = sorted_indices.gather(dim=-1, index=sampled_indices)

            # Compute new beam scores (accumulate log probabilities)
            new_scores = beam_scores[:,
                                     None] + torch.log(probs.gather(dim=-1, index=sampled_tokens))

            # Flatten for beam selection
            new_scores = new_scores.view(-1)
            sampled_tokens = sampled_tokens.view(-1)

            # Select top `num_beams` beams
            top_scores, top_indices = torch.topk(new_scores, num_beams, dim=-1)
            beam_indices = top_indices // probs.shape[-1]  # Original beam indices
            next_tokens = sampled_tokens[top_indices]  # Selected tokens

            # Update token_ids and attention_mask
            token_ids = torch.cat(
                [token_ids[beam_indices],
                 next_tokens.unsqueeze(-1)],
                dim=-1)
            attention_mask = torch.cat(
                [attention_mask[beam_indices],
                 torch.ones_like(next_tokens).unsqueeze(-1)],
                dim=-1)
            beam_scores = top_scores

            # Stop if all beams hit EOS
            if (next_tokens == self.tokenizer.eos_token_id).all():
                break

        # Return the best sequence
        best_index = torch.argmax(beam_scores)
        best_tokens = token_ids[best_index]
        return best_tokens, self.tokenizer.decode(best_tokens.cpu().tolist())[3:]


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


class MultitaskLoader:
    def __init__(self, main_loader, aux_loader, aux_ratio=0.1):
        self.main_loader = iter(main_loader)
        self.aux_loader = cycle(aux_loader)
        self.aux_ratio = aux_ratio
        self.counter = 0
        self.len = len(main_loader)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        main_batch = next(self.main_loader)

        aux_batch = None
        if self.counter % int(1 / self.aux_ratio) == 0:  # Sample aux periodically
            aux_batch = next(self.aux_loader)

        self.counter += 1
        return {'main': main_batch, 'aux': aux_batch}


def paraphrase_task_train(batch, model, device):
    b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten(
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
    return task1_loss, task1_aux_loss


def sonnet_task_train(batch, model, device):
    b_ids, b_mask = batch['token_ids'], batch['attention_mask']
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    output = model(b_ids, b_mask)
    logits = output["next_token_logits"]
    logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
    # Ignore the first token to compose the labels.
    labels = b_ids[:, 1:].contiguous().flatten()
    task2_loss = F.cross_entropy(logits, labels, reduction='mean')
    task2_aux_loss = output["aux_loss"]
    return task2_loss, task2_aux_loss


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

    best_score = 0
    for epoch in range(args.epochs):
        model.train()
        task1_train_loss = 0
        task2_train_loss = 0
        num_batches_main = 0
        num_batches_aux = 0

        combined_loader = MultitaskLoader(
            para_task_dataloaders["train"],
            sonnet_task_dataloaders["train"], aux_ratio=0.1 - epoch * 0.01)
        for batch in tqdm(
                iterable=combined_loader, total=len(combined_loader),
                desc=f"Epoch {epoch + 1} /{args.epochs} "):
            optimizer.zero_grad()

            main_batch = batch['main']
            aux_batch = batch['aux']

            # task 1
            num_batches_main += 1
            task1_loss, task1_aux_loss = paraphrase_task_train(
                main_batch, model, device)
            task1_train_loss += task1_loss.item() + task1_aux_loss.item()

            if aux_batch is not None:
                num_batches_aux += 1
                # task 2
                task2_loss, task2_aux_loss = sonnet_task_train(
                    aux_batch, model, device)
                task2_train_loss += task2_loss.item() + task2_aux_loss.item()

                total_loss = 0.3*(task1_loss + task1_aux_loss) + \
                    0.7 * (task2_loss + task2_aux_loss)
            else:
                total_loss = task1_loss + task1_aux_loss

            total_loss.backward()
            optimizer.step()

            if num_batches_aux != 0 and num_batches_aux % 10 == 0:
                model_eval_sonnet(
                    sonnet_task_dataloaders["dev_held_out"],
                    sonnet_task_dataloaders["dev_label_path"],
                    model, device, args.temperature, args.top_p) / 100

        task1_train_loss = task1_train_loss/num_batches_main
        task2_train_loss = task2_train_loss/num_batches_aux

        para_dev_acc, dev_f1, *_ = model_eval_paraphrase(
            para_task_dataloaders["dev"], model, device)

        sonnet_dev_acc = model_eval_sonnet(
            sonnet_task_dataloaders["dev_held_out"],
            sonnet_task_dataloaders["dev_label_path"],
            model, device, args.temperature, args.top_p) / 100

        # calculate weighted score between tasks
        weighted_score = 0.5 * para_dev_acc + 0.5 * sonnet_dev_acc

        if weighted_score > best_score:
            save_model(model, optimizer, args, args.filepath)
            best_score = weighted_score

        print(
            f"paraphrase_loss: {task1_train_loss:.3f}, sonnet_loss: {task2_train_loss:.3f} para dev acc :: {para_dev_acc:.3f}, sonnet dev acc :: {sonnet_dev_acc:.3f}")
        wandb.log(
            {"paraphrase_loss": task1_train_loss, "sonnet_loss": task2_train_loss,
             "para_dev_acc": para_dev_acc, "sonnet_dev_acc": sonnet_dev_acc,
             "best_score": best_score, "weighted_score": weighted_score,
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
        para_test_data, shuffle=False, batch_size=args.batch_size,
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
    parser.add_argument("--expert_hidden_size", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=2)
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
