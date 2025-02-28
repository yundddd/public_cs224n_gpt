
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
            logits_sequence = self.forward(token_ids, attention_mask)[
                "next_token_logits"]
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
    The only difference is to assign zero weights for tasks that are skipped in the current
    iteration due to dataset imbalance.
    """

    def __init__(self, num_tasks, T=2.0):
        """
        Implements Dynamic Weight Averaging (DWA) for multi-task learning from Liu et al. (2019).

        Args:
            num_tasks (int): Number of tasks.
            T (float): Temperature parameter for weight distribution.
        """
        self.num_tasks = num_tasks
        self.T = T
        self.prev_losses = [1.0] * num_tasks  # Initialize with dummy values

    def get_weights(self, current_losses):
        eps = 1e-8  # Avoid division by zero

        # Only consider tasks with non-zero losses
        valid_indices = [i for i, l in enumerate(current_losses) if l.item() > 0]

        if not valid_indices:  # Handle edge case where all tasks are skipped
            return torch.ones(self.num_tasks) / self.num_tasks

        valid_loss_ratios = [
            self.prev_losses[i] / (current_losses[i].item() + eps)
            for i in valid_indices
        ]

        # Normalize with softmax over valid indices only
        exps = torch.exp(torch.tensor(valid_loss_ratios) / self.T)
        valid_weights = exps / exps.sum()

        # Assign weights to all tasks (zero for skipped tasks)
        weights = torch.zeros(self.num_tasks)
        for idx, w in zip(valid_indices, valid_weights):
            weights[idx] = w

        # Update previous losses only for non-zero tasks
        for i in valid_indices:
            self.prev_losses[i] = max(current_losses[i].item(), eps)

        return weights


def pcgrad_backward(model, losses, optimizer):
    """
    This function implements the PCGrad algorithm from the paper:
    "Gradient Surgery for Multi-Task Learning" (Liu et al
    https://arxiv.org/abs/2001.06782).
    The only modification is to skip gradient shapes that are not compatible.
    For example, different tasks may have different detection heads. However,
    shared backbone like MoE layers can still participate in conflict resolution.
    """
    # Store gradients for each task
    grads = []
    for loss in losses:
        if loss.grad_fn is not None:
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward(retain_graph=True)  # Compute task-specific gradient
            # Save trainable parameter grads
            grads.append([p.grad.clone()
                         for p in model.parameters() if p.grad is not None])

    # Resolve gradient conflicts
    for i in range(len(grads)):
        for j in range(len(grads)):
            if i != j:  # Compare different tasks
                for g_i, g_j in zip(grads[i], grads[j]):
                    # Flatten gradients to ensure compatibility
                    if g_i.shape != g_j.shape:
                        continue  # Skip incompatible gradients

                    # Compute dot product
                    dot_product = torch.sum(g_i * g_j)

                    # Check for conflict (negative dot product)
                    if dot_product < 0:
                        # Resolve conflict by projecting g_i onto the normal plane of g_j
                        projection = (dot_product / (torch.norm(g_j)**2 + 1e-8)) * g_j
                        g_i -= projection

    # Apply resolved gradients
    optimizer.zero_grad()
    for param, *g in zip(model.parameters(), *grads):
        if param.requires_grad:  # Only update trainable parameters
            param.grad = torch.stack(g).mean(dim=0)  # Aggregate modified gradients


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
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
    for epoch in range(args.epochs):
        model.train()
        paraphrase_task_loss, sonnet_train_loss, sentiment_train_loss = 0, 0, 0
        num_batches_paraphrase, num_batches_sonnet, num_batches_sentiment = 0, 0, 0

        combined_loader = MultitaskLoader(
            para_task_dataloaders["train"],
            sonnet_task_dataloaders["train"],
            sentiment_task_dataloaders["train"],
            aux_ratio1=0.0025, aux_ratio2=0.25)
        for batch in tqdm(
                iterable=combined_loader, total=len(combined_loader),
                desc=f"Epoch {epoch + 1} /{args.epochs} "):
            optimizer.zero_grad()

            main_batch = batch['main']
            sonnet_batch = batch['aux1']
            sentiment_batch = batch['aux2']

            # task 1
            paraphrase_loss, paraphrase_aux_loss = paraphrase_task_train(
                main_batch, model, device)
            total_para_loss = paraphrase_loss + paraphrase_aux_loss
            num_batches_paraphrase += 1

            total_sonnet_loss = torch.tensor(0.0, device=device)
            if sonnet_batch is not None:
                sonnet_task_loss, sonnet_aux_loss = sonnet_task_train(
                    sonnet_batch, model, device)
                total_sonnet_loss = sonnet_task_loss + sonnet_aux_loss
                num_batches_sonnet += 1

            total_sentiment_loss = torch.tensor(0.0, device=device)
            if sentiment_batch is not None:
                sentiment_task_loss, sentiment_task_aux_loss = sentiment_task_train(
                    sentiment_batch,
                    model,
                    device)
                total_sentiment_loss = sentiment_task_loss + sentiment_task_aux_loss
                num_batches_sentiment += 1

            current_losses = [total_para_loss, total_sonnet_loss, total_sentiment_loss]

            if args.use_dwa:
                task_weights = dwa.get_weights(current_losses)
                for loss, weight in zip(current_losses, task_weights):
                    loss *= weight

            if args.use_pcgrad:
                pcgrad_backward(model, current_losses, optimizer)
            else:
                total_loss = sum(current_losses)
                total_loss.backward()

            optimizer.step()

            paraphrase_task_loss += total_para_loss.item()
            sonnet_train_loss += total_sonnet_loss.item()
            sentiment_train_loss += total_sentiment_loss.item()

        paraphrase_task_loss = paraphrase_task_loss/num_batches_paraphrase
        sonnet_train_loss = sonnet_train_loss/num_batches_sonnet
        sentiment_train_loss = sentiment_train_loss/num_batches_sentiment

        # evaluate model
        para_dev_acc, para_train_acc, sonnet_dev_acc, sonnet_train_acc, sentiment_dev_acc, sentiment_train_acc = evaluate_model(
            args, model, para_task_dataloaders, sonnet_task_dataloaders, sentiment_task_dataloaders, device)

        # calculate weighted score between tasks
        if para_dev_acc == 0 or sonnet_dev_acc == 0 or sentiment_dev_acc == 0:
            weighted_score = 0
        else:
            weighted_score = 3 / (1 / para_dev_acc + 1 / sonnet_dev_acc + 1 /
                                  sentiment_dev_acc)

        if weighted_score > best_score and not args.debug:
            save_model(model, optimizer, args, args.filepath)
            best_score = weighted_score

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


def evaluate_model(
        args, model, para_task_dataloaders, sonnet_task_dataloaders,
        sentiment_task_dataloaders, device):
    """Evaluate model on all tasks."""
    if args.debug:
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

    parser.add_argument("--weight_decay", type=int, default=0.01)

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
    pass


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-moe.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    dwa = "1dwa" if args.use_dwa else "0dwa"
    pcgrad = "1pcgrad" if args.use_pcgrad else "0pcgrad"

    with wandb.init(
        project="cs224n",
        config=wandb.config,
        name=f"{args.model_size}-{dwa}-{pcgrad}-{args.num_moe_layers}moe-{args.num_experts}exp-{args.expert_hidden_size}eh-{args.aux_loss_weight}aux-{args.weight_decay}wd-{args.lr}lr"
    ):
        wandb.run.log_code(include_fn=lambda path: path.endswith(".py"))
        train(args)
