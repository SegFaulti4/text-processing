import pickle
import numpy as np
import torch
import torch.utils.data
import lzma
import glob
from transformers import AutoTokenizer, BertForTokenClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from tqdm import tqdm

from dataset import TOKENIZER_KWARGS, TEST_TXT_BLACKLIST, TEST_ANN_BLACKLIST, MyCollator, MyDataset


def load_tokenizer(tokenizer_path, tokenizer_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    return tokenizer

def load_model(state_path, config_path):
    with open("state.xz", "rb") as state_f:
        with open("config.xz", "rb") as config_f:
            state_data = state_f.read()
            config_data = config_f.read()

    filters = [{"id": lzma.FILTER_LZMA2, "dict_size": 268435456, "preset": 9, "mf": lzma.MF_HC3, "depth": 0, "lc": 3}]
    state_data = lzma.decompress(state_data, format=lzma.FORMAT_RAW, filters=filters)
    config_data = lzma.decompress(config_data, format=lzma.FORMAT_RAW, filters=filters)

    state = pickle.loads(state_data)
    config = pickle.loads(config_data)
    model = BertForTokenClassification.from_pretrained(config=config, state_dict=state,
                                                       pretrained_model_name_or_path=None)
    return model

def compute_metrics(outputs: torch.Tensor, labels: torch.LongTensor,):
    """
    Compute NER metrics.
    """

    metrics = {}

    y_true = labels[labels != -100].cpu()
    y_pred = torch.argmax(outputs, dim=1)[labels != -100].cpu()

    # accuracy
    accuracy = accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )

    # precision
    precision_micro = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="micro",
        zero_division=0,
    )
    precision_macro = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        zero_division=0,
    )

    # recall
    recall_micro = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="micro",
        zero_division=0,

    )
    recall_macro = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        zero_division=0,
    )
    recall_weighted = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        zero_division=0,
    )

    # f1
    f1_micro = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="micro",
        zero_division=0,
    )
    f1_macro = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        zero_division=0,
    )
    f1_weighted = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["accuracy"] = accuracy

    metrics["precision_micro"]    = precision_micro
    metrics["precision_macro"]    = precision_macro
    metrics["precision_weighted"] = precision_weighted

    metrics["recall_micro"]    = recall_micro
    metrics["recall_macro"]    = recall_macro
    metrics["recall_weighted"] = recall_weighted

    metrics["f1_micro"]    = f1_micro
    metrics["f1_macro"]    = f1_macro
    metrics["f1_weighted"] = f1_weighted

    return metrics


def evaluate(model, dataloader, device):
    model.eval()
    batch_metrics_list = defaultdict(list)

    with torch.no_grad():
        for i, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc="loop over test batches",
        ):
            batch = batch.to(device)
            labels = batch["labels"].cpu()
            # batch.pop("labels")

            outputs = model(**batch)["logits"].transpose(1, 2)
            batch_metrics = compute_metrics(outputs, labels)

            for metric_name, metric_value in batch_metrics.items():
                batch_metrics_list[metric_name].append(metric_value)

        for metric_name, metric_value_list in batch_metrics_list.items():
            metric_value = np.mean(metric_value_list)
            print(f"Test {metric_name}: {metric_value}\n")


def main():
    test_txt_paths = sorted(list(filter(
        lambda x: x not in TEST_TXT_BLACKLIST,
        glob.glob("./data/test/*.txt")
    )))
    test_ann_paths = sorted(list(filter(
        lambda x: x not in TEST_ANN_BLACKLIST,
        glob.glob("./data/test/*.ann")
    )))

    tokenizer_kwargs = TOKENIZER_KWARGS
    tokenizer = load_tokenizer("./tokenizer", tokenizer_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = MyDataset(test_txt_paths, test_ann_paths, split=True)
    collator = MyCollator(tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs, label_padding_value=-100)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )

    model = load_model("./state.xz", "./config.xz").to(device)

    evaluate(model, test_dataloader, device)


if __name__ == "__main__":
    main()
