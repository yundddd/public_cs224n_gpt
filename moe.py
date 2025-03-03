
from datetime import datetime
from models.gpt2_moe import GPT2MoEModel
from optimizer import AdamW
from evaluation import model_eval_paraphrase, model_sentiment_eval, model_test_paraphrase, model_eval_sonnet
from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    SentimentDataset,
    SonnetsDataset,
    load_paraphrase_data,
    load_sentiment_data
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
import signal
import sys
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
        # sentiment classification head
        self.sentiment_classifier = torch.nn.Linear(args.d, args.num_labels)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if not args.full_finetune:
            for param in self.gpt.parameters():
                param.requires_grad = False
            for param in self.gpt.gptmoe_layers.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_token_hidden_state = gpt_output["last_token"]
        paraphrase_logits = self.paraphrase_detection_head(last_token_hidden_state)
        sentiment_logits = self.sentiment_classifier(last_token_hidden_state)

        last_hidden_state = gpt_output["last_hidden_state"]
        next_token_logits = self.gpt.hidden_state_to_token(last_hidden_state)
        return {"paraphrase_logits": paraphrase_logits,
                "sentiment_logits": sentiment_logits,
                "next_token_logits": next_token_logits,
                "aux_loss": gpt_output["aux_loss"]}

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=1, top_p=0.9, top_k=50, max_length=128):
        device = self.get_device()
        token_ids = encoding.to(device)
        attention_mask = torch.ones_like(token_ids, dtype=torch.int64).to(device)

        for _ in range(max_length):
            logits_last_token = self.forward(token_ids, attention_mask)[
                "next_token_logits"][:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            # Sort probabilities and indices
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # Apply top-k filtering
            if top_k > 0:
                sorted_probs[:, top_k:] = 0

            # Apply top-p (nucleus) sampling
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumulative_probs - sorted_probs) < top_p  # Correct mask
            sorted_probs = sorted_probs.masked_fill(~mask, 0.0)

            # Ensure at least one valid token remains
            if sorted_probs.sum(dim=-1).min() == 0:
                sorted_probs[:, 0] = 1

            # Re-normalize and sample
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            sampled_index = torch.multinomial(sorted_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Update token_ids and attention_mask
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(sampled_token)], dim=1)

            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

        generated_output = self.tokenizer.decode(
            token_ids[0].cpu().numpy().tolist(),
            skip_special_tokens=True)
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
    para_train_data = load_paraphrase_data(args.para_train)[:40000]
    para_dev_data = load_paraphrase_data(args.para_dev)[:10000]
    if args.debug:
        para_train_data = para_train_data[:64]
        para_dev_data = para_dev_data[:64]

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

    train_held_out_sonnet_dataset = SonnetsDataset(args.sonnet_train_held_out_path)
    dev_held_out_sonnet_dataset = SonnetsDataset(args.sonnet_dev_held_out_path)

    return {"train": train_sonnet_dataloader,
            "train_held_out_label_path": args.sonnet_train_path,
            "train_held_out": train_held_out_sonnet_dataset,
            "dev_held_out": dev_held_out_sonnet_dataset,
            "dev_label_path": args.sonnet_dev_label_path}


def get_sentiment_task_dataloaders(args):
    train_data1, num_labels1 = load_sentiment_data(args.imdb_train, 'train')
    dev_data1 = load_sentiment_data(args.imdb_dev, 'valid')
    train_data2, num_labels2 = load_sentiment_data(args.sst_train, 'train')
    dev_data2 = load_sentiment_data(args.sst_dev, 'valid')

    if args.debug:
        train_data1 = train_data1[:64]
        dev_data1 = dev_data1[:64]
        train_data2 = train_data2[:64]
        dev_data2 = dev_data2[:64]

    train_dataset1 = SentimentDataset(train_data1, args)
    dev_dataset1 = SentimentDataset(dev_data1, args)
    train_dataset2 = SentimentDataset(train_data2, args)
    dev_dataset2 = SentimentDataset(dev_data2, args)
    train_dataloader1 = DataLoader(
        train_dataset1, shuffle=True, batch_size=args.batch_size,
        collate_fn=train_dataset1.collate_fn)
    dev_dataloader1 = DataLoader(
        dev_dataset1, shuffle=False, batch_size=args.batch_size,
        collate_fn=dev_dataset1.collate_fn)
    train_dataloader2 = DataLoader(
        train_dataset2, shuffle=True, batch_size=args.batch_size,
        collate_fn=train_dataset2.collate_fn)
    dev_dataloader2 = DataLoader(
        dev_dataset2, shuffle=False, batch_size=args.batch_size,
        collate_fn=dev_dataset2.collate_fn)

    combined_train = CombinedDataLoader(train_dataloader1, train_dataloader2)
    combined_dev = CombinedDataLoader(dev_dataloader1, dev_dataloader2)

    return {"train": combined_train, "dev": combined_dev,
            "num_labels": max(num_labels1, num_labels2)}


class CombinedDataLoader:
    def __init__(self, main_loader, aux_loader):
        self.main_loader = main_loader
        self.aux_loader = aux_loader
        self.main_loader_iter = iter(main_loader)
        self.aux_loader_iter = iter(aux_loader)
        self.len = len(main_loader) + len(aux_loader)

    def __iter__(self):
        self.main_loader_iter = iter(self.main_loader)
        self.aux_loader_iter = iter(self.aux_loader)
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        try:
            batch = next(self.main_loader_iter)
        except StopIteration:
            batch = next(self.aux_loader_iter)

        return batch


class MultitaskLoader:
    def __init__(self, main_loader, aux_loader1, aux_loader2, aux_ratio1,
                 aux_ratio2):
        self.main_loader = main_loader
        self.aux_loader1 = aux_loader1
        self.aux_loader2 = aux_loader2

        self.main_loader_iter = iter(self.main_loader)
        self.aux_loader1_iter = cycle(self.aux_loader1)
        self.aux_loader2_iter = cycle(self.aux_loader2)

        self.aux_ratio1 = aux_ratio1
        self.aux_ratio2 = aux_ratio2
        self.counter = 0
        self.len = len(main_loader)

    def __iter__(self):
        self.main_loader_iter = iter(self.main_loader)
        self.aux_loader1_iter = cycle(self.aux_loader1)
        self.aux_loader2_iter = cycle(self.aux_loader2)
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        main_batch = next(self.main_loader_iter)

        aux_batch1 = None
        if self.counter % int(1 / self.aux_ratio1) == 0:  # Sample aux periodically
            aux_batch1 = next(self.aux_loader1_iter)
        aux_batch2 = None
        if self.counter % int(1 / self.aux_ratio2) == 0:  # Sample aux periodically
            aux_batch2 = next(self.aux_loader2_iter)

        self.counter += 1
        return {'main': main_batch, 'aux1': aux_batch1, 'aux2': aux_batch2}


def sentiment_task_train(batch, model, device):
    b_ids, b_mask, b_labels = (batch['token_ids'],
                               batch['attention_mask'], batch['labels'])
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)

    output = model(b_ids, b_mask)
    task_loss = F.cross_entropy(
        output["sentiment_logits"], b_labels.view(-1),
        reduction='mean')
    task_aux_loss = output["aux_loss"]
    return task_loss, task_aux_loss


def paraphrase_task_train(batch, model, device):
    b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten(
    )
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    # Map labels to 0 and 1
    mapped_labels = map_labels(labels).to(device)
    output = model(b_ids, b_mask)
    task_loss = F.cross_entropy(
        output["paraphrase_logits"],
        mapped_labels, reduction='mean')
    task_aux_loss = output["aux_loss"]
    return task_loss, task_aux_loss


def sonnet_task_train(batch, model, device):
    b_ids, b_mask = batch['token_ids'], batch['attention_mask']
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    output = model(b_ids, b_mask)
    logits = output["next_token_logits"]
    logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
    # Ignore the first token to compose the labels.
    labels = b_ids[:, 1:].contiguous().flatten()
    task_loss = F.cross_entropy(logits, labels, reduction='mean')
    task_aux_loss = output["aux_loss"]
    return task_loss, task_aux_loss


class DynamicWeightAveraging:
    """
    Implements Dynamic Weight Averaging (DWA) for multi-task learning from Liu et al. (2019).
    Key changes:
    - Track average of past losses (not just previous loss).
    - Compute loss ratio as current_loss / average_past_loss.
    - Handle initialization correctly.
    """

    def __init__(self, num_tasks, T=2.0):
        self.num_tasks = num_tasks
        self.T = T
        self.avg_losses = None  # Initialize after first iteration
        self.step = 0  # Track number of updates

    def get_weights(self, current_losses):
        eps = 1e-8
        valid_indices = [i for i, l in enumerate(current_losses) if l.item() > 0]

        if not valid_indices:
            return torch.zeros(self.num_tasks)  # Zero weights if all tasks skipped

        # Initialize avg_losses on first call
        if self.avg_losses is None:
            self.avg_losses = [l.item() for l in current_losses]

        # Compute loss ratios for valid tasks
        loss_ratios = []
        for i in valid_indices:
            current_loss = current_losses[i].item()
            avg_past_loss = self.avg_losses[i]
            ratio = current_loss / (avg_past_loss + eps)
            loss_ratios.append(ratio)

        # Compute weights via softmax over valid indices
        exps = torch.exp(torch.tensor(loss_ratios) / self.T)
        valid_weights = exps / exps.sum()

        # Assign weights to all tasks (zero for skipped)
        weights = torch.zeros(self.num_tasks)
        for idx, w in zip(valid_indices, valid_weights):
            weights[idx] = w

        # Update average losses for valid tasks
        self.step += 1
        for i in valid_indices:
            current_loss = current_losses[i].item()
            # EMA-like update: avg = (prev_avg * (step-1) + current_loss) / step
            self.avg_losses[i] = (
                self.avg_losses[i] * (self.step - 1) + current_loss) / self.step

        return weights


def pcgrad_backward(model, losses, optimizer, task_weights=None, scale_factor=1.0):
    """
    Implements PCGrad with DWA weights applied after conflict resolution.

    Args:
        model: The multi-task model.
        losses: List of task losses.
        optimizer: The optimizer.
        task_weights: List of weights from DWA (one per task).
        scale_factor: Scaling factor for gradient projection.
    """
    optimizer.zero_grad()

    # Convert model.parameters() to a list for indexing
    params = list(model.parameters())

    # Collect gradients for ALL parameters, even if unused by a task
    grads = []
    for loss in losses:
        if loss.grad_fn is None:
            # If task is skipped, append zero gradients for all parameters
            grads.append([torch.zeros_like(p) for p in params])
            continue

        loss.backward(retain_graph=True)
        task_grads = []
        for p in params:
            if p.grad is not None:
                task_grads.append(p.grad.clone())
            else:
                # Explicitly track unused parameters with zero gradients
                task_grads.append(torch.zeros_like(p))
        grads.append(task_grads)
        optimizer.zero_grad()

    # Resolve conflicts parameter-wise
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            for param_idx in range(len(params)):
                g_i = grads[i][param_idx]
                g_j = grads[j][param_idx]

                if g_i.shape != g_j.shape:
                    continue  # Skip parameters with mismatched shapes

                dot_product = torch.sum(g_i * g_j)
                if dot_product < 0:
                    projection = (dot_product / (torch.norm(g_j)**2 + 1e-8)) * g_j
                    grads[i][param_idx] -= scale_factor * projection

    # Apply DWA weights to the resolved gradients
    if task_weights is not None:
        if len(task_weights) != len(grads):
            raise ValueError(
                f"task_weights length ({len(task_weights)}) does not match number of tasks ({len(grads)})")

        for task_idx, weight in enumerate(task_weights):
            for param_idx in range(len(params)):
                grads[task_idx][param_idx] *= weight

    # Sum gradients across tasks for each parameter
    for param_idx, param in enumerate(params):
        if param.requires_grad:
            param.grad = sum(task_grads[param_idx] for task_grads in grads)


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # paraphrase task
    para_task_dataloaders = get_paraphrase_task_dataloaders(args)
    sonnet_task_dataloaders = get_sonnet_task_dataloaders(args)
    sentiment_task_dataloaders = get_sentiment_task_dataloaders(args)

    args.num_labels = sentiment_task_dataloaders["num_labels"]
    args = add_arguments(args)
    model = MoEGPT(args)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dwa = DynamicWeightAveraging(num_tasks=3, T=2.0)

    best_score = 0
    patience = 5  # Number of epochs to wait for improvement
    no_improvement_counter = 0  # Counter for epochs without improvement

    for epoch in range(args.epochs):
        model.train()
        paraphrase_task_loss, sonnet_train_loss, sentiment_train_loss = 0, 0, 0
        num_batches = {'para': 0, 'sonnet': 0, 'sentiment': 0}

        combined_loader = MultitaskLoader(
            para_task_dataloaders["train"],
            sonnet_task_dataloaders["train"],
            sentiment_task_dataloaders["train"],
            aux_ratio1=0.0025, aux_ratio2=0.25)
        for batch in tqdm(
                iterable=combined_loader, total=len(combined_loader),
                desc=f"Epoch {epoch + 1} /{args.epochs} "):
            optimizer.zero_grad()

            # Compute losses for active tasks
            current_losses = []
            active_tasks = []

            # --- Paraphrase task ---
            if 'main' in batch:
                loss, aux_loss = paraphrase_task_train(batch['main'], model, device)
                total_para_loss = loss + aux_loss
                current_losses.append(total_para_loss)
                active_tasks.append(0)
                num_batches['para'] += 1
            else:
                current_losses.append(torch.tensor(0.0, device=device))

            # --- Sonnet task ---
            if 'aux1' in batch and batch['aux1'] is not None:
                loss, aux_loss = sonnet_task_train(batch['aux1'], model, device)
                total_sonnet_loss = loss + aux_loss
                current_losses.append(total_sonnet_loss)
                active_tasks.append(1)
                num_batches['sonnet'] += 1
            else:
                current_losses.append(torch.tensor(0.0, device=device))

            # --- Sentiment task ---
            if 'aux2' in batch and batch['aux2'] is not None:
                loss, aux_loss = sentiment_task_train(batch['aux2'], model, device)
                total_sentiment_loss = loss + aux_loss
                current_losses.append(total_sentiment_loss)
                active_tasks.append(2)
                num_batches['sentiment'] += 1
            else:
                current_losses.append(torch.tensor(0.0, device=device))

            # Get DWA weights (only for active tasks)
            if args.use_dwa:
                task_weights = dwa.get_weights(current_losses)
            else:
                task_weights = torch.ones(3, device=device)

            # Apply PCGrad or standard backward
            if args.use_pcgrad:
                pcgrad_backward(model, current_losses, optimizer, task_weights)
            else:
                total_loss = sum(loss * weight for loss,
                                 weight in zip(current_losses, task_weights))
                total_loss.backward()

            optimizer.step()

            # Log losses
            paraphrase_task_loss += current_losses[0].item()
            sonnet_train_loss += current_losses[1].item()
            sentiment_train_loss += current_losses[2].item()

        # Average losses
        paraphrase_task_loss /= num_batches['para'] if num_batches['para'] > 0 else 1
        sonnet_train_loss /= num_batches['sonnet'] if num_batches['sonnet'] > 0 else 1
        sentiment_train_loss /= num_batches['sentiment'] if num_batches['sentiment'] > 0 else 1

        # evaluate model
        para_dev_acc, para_train_acc, sonnet_dev_acc, sonnet_train_acc, sentiment_dev_acc, sentiment_train_acc = evaluate_model(
            args, model, para_task_dataloaders, sonnet_task_dataloaders, sentiment_task_dataloaders, device)

        # calculate weighted score between tasks
        if para_dev_acc == 0 or sonnet_dev_acc == 0 or sentiment_dev_acc == 0:
            weighted_score = 0
        else:
            weighted_score = 3 / (1 / para_dev_acc + 1 / sonnet_dev_acc + 1 /
                                  sentiment_dev_acc)

        # Early termination logic
        if weighted_score > best_score:
            print(
                f"Validation score improved from {best_score:.3f} to {weighted_score:.3f}")
            best_score = weighted_score
            no_improvement_counter = 0  # Reset patience counter
        else:
            no_improvement_counter += 1
            print(
                f"No improvement in validation score for {no_improvement_counter} epochs (best: {best_score:.3f})")

        print(
            f"paraphrase_loss: {paraphrase_task_loss:.3f}, sonnet_loss: {sonnet_train_loss:.3f}, sentiment_loss: {sentiment_train_loss:.3f} para dev acc :: {para_dev_acc:.3f}, sonnet dev acc :: {sonnet_dev_acc:.3f}, sentiment dev acc :: {sentiment_dev_acc:.3f}")
        if not args.debug:
            wandb.log(
                {"paraphrase_loss": paraphrase_task_loss, "sonnet_loss": sonnet_train_loss,
                 "sentiment_loss": sentiment_train_loss, "para_dev_acc": para_dev_acc,
                 "sonnet_dev_acc": sonnet_dev_acc, "sentiment_dev_acc": sentiment_dev_acc,
                 "para_train_acc": para_train_acc, "sonnet_train_acc": sonnet_train_acc,
                 "sentiment_train_acc": sentiment_train_acc, "best_score": best_score,
                 "weighted_score": weighted_score, "epoch": epoch})

         # Stop training if patience is exhausted
        if no_improvement_counter >= patience:
            print(f"Early termination: No improvement for {patience} epochs.")
            return


def evaluate_model(
        args, model, para_task_dataloaders, sonnet_task_dataloaders,
        sentiment_task_dataloaders, device):
    """Evaluate model on all tasks."""
    if args.debug or args.no_eval:
        return 0, 0, 0, 0, 0, 0  # Skip evaluation in debug mode
    para_dev_acc, *_ = model_eval_paraphrase(
        para_task_dataloaders["dev"],
        model, device, mode="dev")
    para_train_acc, *_ = model_eval_paraphrase(
        para_task_dataloaders["train"],
        model, device, mode="train")

    sonnet_dev_acc = model_eval_sonnet(
        sonnet_task_dataloaders["dev_held_out"],
        sonnet_task_dataloaders["dev_label_path"],
        model, device, args.temperature, args.top_p, mode="dev") / 100
    sonnet_train_acc = model_eval_sonnet(
        sonnet_task_dataloaders["train_held_out"],
        sonnet_task_dataloaders["train_held_out_label_path"],
        model, device, args.temperature, args.top_p, mode="train") / 100

    sentiment_dev_acc, *_ = model_sentiment_eval(
        sentiment_task_dataloaders["dev"],
        model, device, mode="dev")
    sentiment_train_acc, *_ = model_sentiment_eval(
        sentiment_task_dataloaders["train"],
        model, device, mode="train")

    return para_dev_acc, para_train_acc, sonnet_dev_acc, sonnet_train_acc, sentiment_dev_acc, sentiment_train_acc


@torch.no_grad()
def test_paraphrase(args):
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


@torch.no_grad()
def test_sentiment(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)

    model = MoEGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    dataloader = get_sentiment_task_dataloaders(args)

    test_pred, test_sents, test_sent_ids = model_test_eval(
        dataloader["test"], model, device)
    print('DONE Test')

    with open(args.test_out, "w+") as f:
        f.write(f"id \t Predicted_Sentiment \n")
        for p, s in zip(test_sent_ids, test_pred):
            f.write(f"{p}, {s} \n")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imdb_train", type=str, default="data/ids-cfimdb-train.csv")
    parser.add_argument("--imdb_dev", type=str, default="data/ids-cfimdb-dev.csv")
    parser.add_argument("--imdb_test", type=str,
                        default="data/ids-cfimdb-test-student.csv")

    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str,
                        default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str,
                        default="predictions/para-test-output.csv")

    parser.add_argument("--sonnet_train_path", type=str,
                        default="data/sonnets_train.txt")
    parser.add_argument("--sonnet_train_held_out_path", type=str,
                        default="data/sonnets_train_held_out.txt")
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no_eval", action='store_true')
    parser.add_argument("--sweep", action='store_true')

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int,
        default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size", type=str,
        help="The model size as specified on hugging face. DO NOT use the xl model.",
        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
        default='gpt2')

    parser.add_argument("--use_dwa", action='store_true')
    parser.add_argument("--use_pcgrad", action='store_true')
    parser.add_argument("--full_finetune", action='store_true')

    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--num_moe_layers", type=int, default=1)
    parser.add_argument("--expert_hidden_size", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=3)
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


def add_arg_from_wandb_config(args, config):
    args.lr = config.lr
    args.num_moe_layers = config.num_moe_layers
    args.expert_hidden_size = config.expert_hidden_size
    args.num_experts = config.num_experts
    args.aux_loss_weight = config.aux_loss_weight
    args.weight_decay = config.weight_decay
    args.use_dwa = config.use_dwa
    args.use_pcgrad = config.use_pcgrad
    args.model_size = config.model_size
    args.batch_size = config.batch_size
    args.epochs = config.epochs

    return args


def make_sweep_config():
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'lr': {
                'values': [1.5e-4, 1e-3]
            },
            'num_moe_layers': {
                'values': [1, 2]
            },
            'expert_hidden_size': {
                'values': [32, 128]
            },
            'num_experts': {
                'values': [2, 3, 4]
            },
            'aux_loss_weight': {
                'values': [0.001, 0.01]
            },
            'weight_decay': {
                'values': [0, 1]
            },
            'use_dwa': {
                'values': [True, False]
            },
            'use_pcgrad': {
                'values': [True, False]
            },
            'model_size': {
                'values': ['gpt2', 'gpt2-medium']
            },
            'batch_size': {
                'values': [18]
            },
            'epochs': {
                'values': [15]
            },
        }
    }
    metric = {
        'name': 'weighted_score',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric
    return sweep_config


def train_wrapper(args):
    with wandb.init(
        project="cs224n",
        config=wandb.config,
    ):
        args = add_arg_from_wandb_config(args, wandb.config)

        dwa = "1dwa" if args.use_dwa else "0dwa"
        pcgrad = "1pcgrad" if args.use_pcgrad else "0pcgrad"
        wandb.run.name = (
            f"{args.model_size}-{dwa}-{pcgrad}-"
            f"{args.num_moe_layers}moe-{args.num_experts}exp-"
            f"{args.expert_hidden_size}eh-{args.aux_loss_weight}aux-"
            f"{args.weight_decay}wd-{args.lr}lr"
        )
        print("=======args========")
        print(args)
        wandb.run.log_code(include_fn=lambda path: path.endswith(".py"))
        train(args)


def signal_handler(sig, frame):
    print("Run interrupted! Marking as failed.")
    wandb.finish(exit_code=1)  # Ensures W&B logs it as failed
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

if __name__ == "__main__":
    ### new sweep ###
    # sweep_config = make_sweep_config()
    # sweep_id = wandb.sweep(sweep_config, project="cs224n")
    # sys.exit(0)

    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-moe.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    if args.sweep:
        wandb.agent("cs224n/1bp0douk", function=lambda: train_wrapper(args))

    else:
        with wandb.init(
            project="cs224n",
            config=args,
        ):
            dwa = "1dwa" if args.use_dwa else "0dwa"
            pcgrad = "1pcgrad" if args.use_pcgrad else "0pcgrad"
            wandb.run.name = (
                f"{args.model_size}-{dwa}-{pcgrad}-"
                f"{args.num_moe_layers}moe-{args.num_experts}exp-"
                f"{args.expert_hidden_size}eh-{args.aux_loss_weight}aux-"
                f"{args.weight_decay}wd-{args.lr}lr"
            )
            wandb.run.log_code(include_fn=lambda path: path.endswith(".py"))
            train(args)
