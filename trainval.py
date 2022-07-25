import tqdm, os
import nltk
import pandas as pd
import numpy as np
from datasets import load_metric

from datasets import Dataset
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from haven import haven_wizard as hw
from haven import haven_utils as hu

DEVICE = "cuda"


def tokenize(tokenizer, dataset, input_len=1024):
    def _tokenize(dataset):
        # tokenize documents
        model_inputs = tokenizer(
            dataset["document"],
            max_length=input_len,
            truncation=True,
            return_tensors=None,
        )
        # tokenize summaries
        with tokenizer.as_target_tokenizer():
            model_inputs["labels"] = tokenizer(
                dataset["summary"], max_length=input_len, truncation=True
            )["input_ids"]
        return model_inputs

    t_func = lambda data: _tokenize(data)
    dataset = dataset.map(t_func, batched=True)
    dataset = dataset.remove_columns("document")
    dataset = dataset.remove_columns("summary")

    return dataset


def train_on_loader(model, optim, train_loader):
    model.train()

    for batch in tqdm.tqdm(train_loader, desc="training"):
        # Get Input
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
        )

        # Compute Loss and Update
        optim.zero_grad()
        loss = outputs["loss"]
        loss.backward()
        optim.step()

    return {"train_loss": float(loss)}


def val_on_loader(model, tokenizer, val_loader):
    metric = load_metric("rouge")

    model.eval()
    for batch in tqdm.tqdm(val_loader, desc="validating"):
        # Get Input
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
        )

        pred_tokens = outputs["logits"].argmax(dim=2)
        preds = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        labels_cleaned = np.where(
            labels.cpu() != -100, labels.cpu(), tokenizer.pad_token_id
        )
        gt = tokenizer.batch_decode(labels_cleaned, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in gt]

        rouge1 = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )["rouge1"].high.fmeasure
        rouge2 = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )["rouge2"].high.fmeasure

    return {"val_rouge1": float(rouge1), "val_rouge2": float(rouge2)}


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Get dataset
    train_set = Dataset.from_dict(hu.load_json("data/train.json"))
    val_set = Dataset.from_dict(hu.load_json("data/val.json"))
    # print(dataset)

    # Get Tokenizer and Model
    model_checkpoint = exp_dict["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).cuda()
    optim = AdamW(model.parameters(), lr=5e-5)

    # Tokenize dataset
    train_set = tokenize(tokenizer, train_set, input_len=1024)
    val_set = tokenize(tokenizer, val_set, input_len=1024)

    # Get Loaders
    collate_fn = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_loader = DataLoader(
        train_set, collate_fn=collate_fn, batch_size=16, shuffle=True
    )
    val_loader = DataLoader(val_set, collate_fn=collate_fn, batch_size=16)

    # Train and Validate
    score_list = []
    for epoch in range(10):
        score_dict = {"epoch": epoch}

        # Val for one epoch
        val_dict = val_on_loader(model, tokenizer, val_loader)
        # Train for one epoch
        train_dict = train_on_loader(model, optim, train_loader)

        # Get Metrics
        score_dict.update(train_dict)
        score_dict.update(val_dict)

        # Save Metrics in "savedir" as score_list.pkl
        score_list += [score_dict]
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(os.path.join(savedir, "score_list.pkl"), score_list)

    print("Experiment done\n")


if __name__ == "__main__":
    # Define a list of experiments
    exp_list = []

    # Assignment: Add more models here
    for model in [
        "TheLongSentance/t5-small-finetuned-xsum",
        "sshleifer/distilbart-cnn-6-6",
    ]:
        exp_list += [{"model": model}]

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_list,
        results_fname="results.ipynb",
        savedir_base="results",
        reset=True,
    )
"""
if 1:
    data = hu.load_json("data/twitter.json")
    n = 320
    train = {"summary": data["summary"][:n], "document": data["document"][:n]}
    val = {
        "summary": data["summary"][n : n + 32],
        "document": data["document"][n : n + 32],
    }
    hu.save_json("data/train.json", train)
    hu.save_json("data/val.json", val)
"""
