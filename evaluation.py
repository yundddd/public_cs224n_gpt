# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
    SonnetsDataset,
)

TQDM_DISABLE = False


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
    def map_labels(labels):
        label_map = {3919: 1, 8505: 0}
        return torch.tensor(
            [label_map[label.item()] for label in labels],
            dtype=torch.long)

    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true, y_pred, sent_ids = [], [], []
    for step, batch in enumerate(
        tqdm(
            dataloader, desc=f'paraphrase dev eval',
            disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
            'labels'].flatten()

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        output = model(b_ids, b_mask)
        if isinstance(output, dict):
            logits = output['classification_logits'].cpu().numpy()
        else:
            logits = model(b_ids, b_mask).cpu().numpy()
        preds = logits.flatten()

        # mapped_labels = map_labels(labels)
        y_true.extend(labels)
        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true, y_pred, sent_ids = [], [], []
    for step, batch in enumerate(
        tqdm(
            dataloader, desc=f'paraphrase test eval',
            disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        output = model(b_ids, b_mask)
        if isinstance(output, dict):
            logits = output['classification_logits'].cpu().numpy()
        else:
            logits = model(b_ids, b_mask).cpu().numpy()
        preds = logits.flatten()

        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

    return y_pred, sent_ids


@torch.no_grad()
def model_eval_sonnet(
    dev_held_out_dataset, dev_label_path, model, device, temperature,
        top_p):
    model.eval()

    generated_sonnets = []
    for batch in tqdm(dev_held_out_dataset, desc='sonnet dev eval',
                      disable=TQDM_DISABLE):
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1],
            return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(
            encoding['input_ids'],
            temperature=temperature, top_p=top_p)[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))

        # print(f'{decoded_output}\n\n')

    with open("/tmp/sonnet_dev_completion", "w+") as f:
        f.write("--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])

    return test_sonnet("/tmp/sonnet_dev_completion", dev_label_path)


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path, quite=True)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path, quite=True)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)
