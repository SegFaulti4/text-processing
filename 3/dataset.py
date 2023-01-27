from typing import List, Tuple

import torch
import torch.utils.data
from transformers.tokenization_utils_base import BatchEncoding


TEST_BLACKLIST = [
    "165459_text",
    "176167_text",
    "178485_text",
    "192238_text",
    "193267_text",
    "193946_text",
    "194112_text",
    "2021",
    "202294_text",
    "2031",
    "209438_text",
    "209731_text",
    "546860_text"
]
TEST_TXT_BLACKLIST = [
    "./data/test/" + b + ".txt" for b in TEST_BLACKLIST
]
TEST_ANN_BLACKLIST = [
    "./data/test/" + b + ".ann" for b in TEST_BLACKLIST
]
TOKENIZER_KWARGS = {
    "is_split_into_words": False,
    "return_offsets_mapping": True,
    "padding": True,
    "truncation": True,
    "max_length": 5000,
    "return_tensors": "pt",
}


class MyDataset(torch.utils.data.Dataset):

    Entity = Tuple[str, int, int]
    Item = Tuple[str, List[Entity]]

    def __init__(self, txt_paths: List[str], ann_paths: List[str], split: bool):
        """
        txt_paths   - список путей до файлов с исходным текстом
        ann_paths   - список путей до файлов с аннотациями
        split       - делить ли текст на параграфы
        """
        self.texts = []
        self.annotations = []

        for txt_path, ann_path in zip(txt_paths, ann_paths):
            txt = self.read_txt(txt_path)
            ann = self.read_ann(ann_path)
            if split:
                texts, annotations = self.split_paragraphs(txt, ann)
                self.texts.extend(texts)
                self.annotations.extend(annotations)
            else:
                self.texts.append(txt)
                self.annotations.append(ann)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Item:
        """
        returns: параграф текста и список аннотаций к нему,
                 либо весь текст целиком и список его аннотаций,
                 если параметр split был равен False
        """
        return self.texts[idx], self.annotations[idx]

    @staticmethod
    def read_txt(txt_path: str) -> str:
        return open(txt_path, "r").read() # "".join(open(txt_path, "r").readlines())

    @staticmethod
    def read_ann(ann_path: str) -> List[Entity]:
        entities = []
        with open(ann_path, "r") as annF:
            for entity_line in filter(lambda x: x.startswith("T"), annF.readlines()):
                if ';' in entity_line:
                    pass
                else:
                    parts = entity_line.strip().split()
                    name, start, end = parts[1:4]
                    entities.append((name, int(start), int(end)))

        res = []
        entities = sorted(entities, key=lambda x: (x[1], -x[2]))
        for i, ent in enumerate(entities):
            if i + 1 < len(entities):
                next_ent = entities[i + 1]
                if next_ent[1] >= ent[2]:
                    res.append(ent)
            else:
                res.append(ent)

        return res

    @staticmethod
    def split_paragraphs(txt: str, entities: List[Entity]) -> Tuple[List[str], List[List[Entity]]]:
        texts = []
        annotations = []

        txt_start = 0
        entities_start = 0
        par_pos = txt.find("\n\n")
        while par_pos != -1:
            par = txt[txt_start:par_pos]
            par_entities = []
            for i, entity in enumerate(entities[entities_start:]):
                if entity[1] > par_pos:
                    entities_start += i
                    break
                new_entity = entity[0], entity[1] - txt_start, entity[2] - txt_start

                par_entities.append(new_entity)

            texts.append(par)
            annotations.append(par_entities)

            txt_start = par_pos + 2
            par_pos = txt.find("\n\n", txt_start)
            if par_pos != -1:
                while par_pos + 2 < len(txt) and txt[par_pos + 2] == '\n':
                    par_pos += 1

        return texts, annotations


class MyCollator:

    ENTITY_TYPES = [
        "AGE",
        "AWARD",
        "CITY",
        "COUNTRY",
        "CRIME",
        "DATE",
        "DISEASE",
        "DISTRICT",
        "EVENT",
        "FACILITY",
        "FAMILY",
        "IDEOLOGY",
        "LANGUAGE",
        "LAW",
        "LOCATION",
        "MONEY",
        "NATIONALITY",
        "NUMBER",
        "ORDINAL",
        "ORGANIZATION",
        "PENALTY",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "PROFESSION",
        "RELIGION",
        "STATE_OR_PROV",
        "TIME",
        "WORK_OF_ART"
    ]

    LABELS = [
        "B-" + t for t in ENTITY_TYPES
    ] + [
        "I-" + t for t in ENTITY_TYPES
    ] + [
        "O"
    ]

    UNKNOWN_LABEL = "<UNK>"

    NUM_LABELS = len(LABELS) + 1

    LABEL2ID = {
        label: i for i, label in enumerate(LABELS)
    }

    def __init__(self, tokenizer, tokenizer_kwargs, label_padding_value: int):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

        self.LABEL2ID[self.UNKNOWN_LABEL] = label_padding_value

    def __call__(self, batch: List[MyDataset.Item]):
        texts, annotations = zip(*batch)
        tokens = self.tokenizer(texts, **self.tokenizer_kwargs)
        label_ids = self.encode_annotations(tokens, annotations)
        tokens.pop("offset_mapping")

        tokens["labels"] = label_ids
        return tokens

    @staticmethod
    def encode_annotations(
        tokens: BatchEncoding,
        annotations: List[List[MyDataset.Entity]]
    ) -> torch.LongTensor:

        res = list()
        for doc_ann, doc_offset in zip(annotations, tokens.offset_mapping):
            res.append(list())
            offset_start = 0
            for entity in doc_ann:

                start_div = 0
                # handle tokens outside of entity
                for i, offset in enumerate(doc_offset[offset_start:]):
                    if offset[0] == offset[1]:
                        start_div = i + 1
                        res[-1].append(MyCollator.UNKNOWN_LABEL)
                    elif offset[0] < entity[1]:
                        start_div = i + 1
                        res[-1].append("O")
                    else:
                        start_div = i
                        break
                offset_start += start_div

                # handle first token from entity
                res[-1].append(f"B-{entity[0]}")
                offset_start += 1

                start_div = 0
                # handle last tokens from entity
                for i, offset in enumerate(doc_offset[offset_start:]):
                    if entity[1] <= offset[0] and offset[1] <= entity[2]:
                        start_div = i + 1
                        res[-1].append(f"I-{entity[0]}")
                    else:
                        start_div = i
                        break
                offset_start += start_div

            # handle the rest of tokens
            for offset in doc_offset[offset_start:]:
                if offset[0] == offset[1]:
                    res[-1].append(MyCollator.UNKNOWN_LABEL)
                else:
                    res[-1].append("O")

            res[-1] = MyCollator.labels2ids(res[-1])

            if len(res[-1]) != len(doc_offset):
                pass

        result = torch.LongTensor(res)
        res.clear()
        return result

    @staticmethod
    def labels2ids(labels: List[str]) -> List[int]:
        return [MyCollator.LABEL2ID[label]
                if label in MyCollator.LABEL2ID
                else MyCollator.LABEL2ID[MyCollator.UNKNOWN_LABEL]
                for label in labels]

    def interpret(self, txt: str, answer: List[int]):
        token_offsets = self.tokenizer(txt, **self.tokenizer_kwargs).offset_mapping.cpu()[0].tolist()

        entities = set()
        idx = 0

        while idx < len(answer):
            label_id = answer[idx]
            label = "O" if label_id == -100 else self.LABELS[label_id]
            # print(idx, answer[idx], label)

            if label[0] != "B":
                idx += 1
                continue

            entity_name = label[2:]
            entity_start = token_offsets[idx][0]
            entity_end = token_offsets[idx][1]

            new_idx = idx + 1
            for label_id, token_offset in zip(answer[idx + 1:], token_offsets[idx + 1:]):
                label = "O" if label_id == -100 else self.LABELS[label_id]

                if label != f"I-{entity_name}":
                    break

                entity_end = token_offset[1]
                new_idx += 1

            idx = new_idx
            entities.add((entity_start, entity_end, entity_name))

        return entities

    def tokenize(self, txt: str):
        tokens = self.tokenizer(txt, **self.tokenizer_kwargs)
        tokens.pop("offset_mapping")
        return tokens
