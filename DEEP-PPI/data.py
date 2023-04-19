import torch
from torch.utils.data import Dataset


class CustomStarDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self, dataframe, tokenizer):
        # load dataframe
        self.df = dataframe
        self.df["label"] = self.df["label"].apply(lambda x: 1 if x != "negative" else 0)
        # extract labels
        self.labels = torch.tensor(self.df["label"].to_numpy().reshape(-1)).long()

        # tokenize sequences
        self.seq1, self.attention_mask1 = self.tokenize_sequences(
            self.df["seq1"].to_list(), tokenizer
        )
        self.seq2, self.attention_mask2 = self.tokenize_sequences(
            self.df["seq2"].to_list(), tokenizer
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.seq1[idx],
            self.seq2[idx],
            self.labels[idx],
            self.attention_mask1[idx],
            self.attention_mask2[idx],
        )

    def tokenize_sequences(self, sequences, tokenizer):
        tokenized_sentences = tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized_sentences["input_ids"]
        attention_mask = tokenized_sentences["attention_mask"]
        return input_ids, attention_mask
