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
def model_eval_paraphrase(dataloader, model, device, mode):
    def map_labels(labels):
        label_map = {3919: 1, 8505: 0}
        return torch.tensor(
            [label_map[label.item()] for label in labels],
            dtype=torch.long)

    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true, y_pred, sent_ids = [], [], []
    expert_layer_counts = None
    for step, batch in enumerate(
        tqdm(
            dataloader, desc=f'paraphrase {mode} eval',
            disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
            'labels'].flatten()

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        output = model(b_ids, b_mask)
        if isinstance(output, dict):
            logits = output['paraphrase_logits'].cpu().numpy()
        else:
            logits = model(b_ids, b_mask).cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        mapped_labels = map_labels(labels)
        y_true.extend(mapped_labels)
        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

        if expert_layer_counts is None:
            expert_layer_counts = output["expert_layer_count"]
        else:
            expert_layer_counts += output["expert_layer_count"]

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sent_ids, expert_layer_counts


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
            logits = output['paraphrase_logits'].cpu().numpy()
        else:
            logits = model(b_ids, b_mask).cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sent_ids.extend(b_sent_ids)

    return y_pred, sent_ids


@torch.no_grad()
def model_eval_sonnet(
    dev_held_out_dataset, dev_label_path, model, device, temperature,
        top_p, mode):
    model.eval()

    generated_sonnets = []
    count = 0
    expert_layer_counts = None

    for batch in tqdm(dev_held_out_dataset, desc=f'sonnet {mode} eval',
                      disable=TQDM_DISABLE):
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1],
            return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(
            encoding['input_ids'],
            temperature=temperature, top_p=top_p)

        decoded_output = model.tokenizer.decode(output[0][0])
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))

        if count < 1:
            print(f'{decoded_output}\n\n')
            count += 1
        if expert_layer_counts is None:
            expert_layer_counts = output[2]
        else:
            expert_layer_counts += output[2]

    with open("/tmp/sonnet_dev_completion", "w+") as f:
        f.write("--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])

    return test_sonnet(
        "/tmp/sonnet_dev_completion", dev_label_path), expert_layer_counts


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


def model_sentiment_eval(dataloader, model, device, mode):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    expert_layer_counts = None

    for step, batch in enumerate(
        tqdm(
            dataloader, desc=f'sentiment {mode} eval',
            disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch[
            'attention_mask'], batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        output = model(b_ids, b_mask)
        logits = output['sentiment_logits']
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)
        if expert_layer_counts is None:
            expert_layer_counts = output["expert_layer_count"]
        else:
            expert_layer_counts += output["expert_layer_count"]

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids, expert_layer_counts
