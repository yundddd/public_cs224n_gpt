# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        # labels = torch.LongTensor([x[2] for x in all_data])
        labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
        labels = self.tokenizer(labels, return_tensors='pt',
                                padding=True, truncation=True)['input_ids']
        sent_ids = [x[3] for x in all_data]

        cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                             (s1, s2) in zip(sent1, sent2)]
        encoding = self.tokenizer(
            cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sent_ids': sent_ids
        }

        return batched_data


class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]

        cloze_style_sents = [
            f'Is "{s1} " a paraphrase of "{s2} "? Answer "yes" or "no": '
            for (s1, s2) in zip(sent1, sent2)]

        encoding = self.tokenizer(
            cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': sent_ids
        }

        return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append(
                        (preprocess_string(record['sentence1']),
                         preprocess_string(record['sentence2']),
                         int(float(record['is_duplicate'])),
                         sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
    return paraphrase_data


class SonnetsDataset(Dataset):
    def __init__(self, file_path, quite=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sonnets = self._load_sonnets(file_path)
        if not quite:
            print(f"loaded {len(self.sonnets)} sonnets from {file_path}")

    def _load_sonnets(self, file_path):
        """Reads the file and extracts individual sonnets."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
        sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

        # Strip leading/trailing spaces
        return [s.strip() for s in sonnets]

    def __len__(self):
        return len(self.sonnets)

    def __getitem__(self, idx):
        return (idx, self.sonnets[idx])

    def collate_fn(self, all_data):
        idx = [example[0] for example in all_data]
        sonnets = [example[1] for example in all_data]

        encoding = self.tokenizer(sonnets, return_tensors='pt',
                                  padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': idx
        }

        return batched_data

class PairwiseSonnetsDataset(Dataset):
    def __init__(self, hq_file_path, lq_file_path, quiet=False):
        """
        Loads high-quality and low-quality sonnets, ensuring alignment.
        HQ sonnets provide "better" completions, LQ sonnets provide "worse" completions.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load high-quality and low-quality sonnets
        self.hq_sonnets = self._load_sonnets(hq_file_path)
        self.lq_sonnets = self._load_sonnets(lq_file_path)
        assert len(self.hq_sonnets) == len(self.lq_sonnets), \
            "Mismatch: HQ and LQ datasets must have the same number of sonnets!"

        if not quiet:
            print(f"Loaded {len(self.hq_sonnets)} sonnets from {hq_file_path} (HQ)")
            print(f"Loaded {len(self.lq_sonnets)} sonnets from {lq_file_path} (LQ)")

    def _load_sonnets(self, file_path):
        """Reads the file and extracts individual sonnets."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
        sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text
        return [s.strip() for s in sonnets]

    def __len__(self):
        return len(self.hq_sonnets)

    def __getitem__(self, idx):
        """Return the prompt (first 3 lines), better completion (HQ), and worse completion (LQ)."""
        hq_sonnet = self.hq_sonnets[idx].split("\n")  # Split into lines
        lq_sonnet = self.lq_sonnets[idx].split("\n")  # Split into lines
        # Extract first 3 lines as the "prompt"
        prompt = "\n".join(hq_sonnet[:3])  # Use HQ prompt (first 3 lines)

        # Extract remaining lines as completions
        better_completion = "\n".join(hq_sonnet[3:])  # True Shakespearean continuation
        worse_completion = "\n".join(lq_sonnet[3:])  # Lower-quality generated completion

        return {
            "prompt": prompt,
            "better": better_completion,
            "worse": worse_completion
        }

    def collate_fn(self, all_data):
        """Custom collate function to batch process prompts and completions with enforced LongTensor."""
        prompts = [example["prompt"] for example in all_data]
        better_completions = [example["better"] for example in all_data]
        worse_completions = [example["worse"] for example in all_data]
        encoding_prompts = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        encoding_better = self.tokenizer(better_completions, return_tensors='pt', padding=True, truncation=True)
        encoding_worse = self.tokenizer(worse_completions, return_tensors='pt', padding=True, truncation=True)
        # Ensure token IDs are LongTensor
        batched_data = {
            "prompt_ids": torch.LongTensor(encoding_prompts["input_ids"]),
            "prompt_mask": torch.LongTensor(encoding_prompts["attention_mask"]),
            "better_ids": torch.LongTensor(encoding_better["input_ids"]),
            "better_mask": torch.LongTensor(encoding_better["attention_mask"]),
            "worse_ids": torch.LongTensor(encoding_worse["input_ids"]),
            "worse_mask": torch.LongTensor(encoding_worse["attention_mask"])
        }

        return batched_data
