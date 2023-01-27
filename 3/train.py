import glob

import torch
import pickle
import lzma
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

from dataset import MyDataset, MyCollator, TOKENIZER_KWARGS


def main():
    model_name = "cointegrated/rubert-tiny2"
    tokenizer_kwargs = TOKENIZER_KWARGS
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", **tokenizer_kwargs)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    train_txt_paths = sorted(list(
        glob.glob("./data/train/*.txt")
    ))
    train_ann_paths = sorted(list(
        glob.glob("./data/train/*.ann")
    ))

    valid_txt_paths = sorted(list(
        glob.glob("./data/dev/*.txt")
    ))
    valid_ann_paths = sorted(list(
        glob.glob("./data/dev/*.ann")
    ))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = MyDataset(train_txt_paths, train_ann_paths, split=True)
    valid_dataset = MyDataset(valid_txt_paths, valid_ann_paths, split=True)
    collator = MyCollator(tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs, label_padding_value=-100)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=MyCollator.NUM_LABELS
    ).to(device)

    args = TrainingArguments(
        output_dir="./working",
        learning_rate=8e-5,
        weight_decay=1e-3,
        lr_scheduler_type='cosine',
        full_determinism=False,
        seed=1337,
        per_device_train_batch_size=8,
        num_train_epochs=50,
        evaluation_strategy='steps',
        eval_steps=5000,
        save_steps=5000
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    trainer.train()

    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            return model

    model_final = unwrap_model(trainer.model_wrapped)
    model_final = model_final.to("cpu")

    def compress_data(data):
        return lzma.compress(
            pickle.dumps(data),
            format=lzma.FORMAT_RAW,
            filters=[{
                "id": lzma.FILTER_LZMA2,
                "dict_size": 268435456,
                "preset": 9,
                "mf": lzma.MF_HC3,
                "depth": 0,
                "lc": 3
            }])

    compressed_state = compress_data(model_final.state_dict())
    compressed_config = compress_data(model_final.config)

    outF = open("state.xz", "wb")
    outF.write(compressed_state)
    outF.close()

    outF = open("config.xz", "wb")
    outF.write(compressed_config)
    outF.close()


if __name__ == "__main__":
    main()
