import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
from torch import nn, optim
from transformers import AutoModelForMaskedLM


class BaseLineModel(pl.LightningModule):
    def __init__(
        self, embedding_dim=320, hidden_dim=128, base_model="facebook/esm2_t6_8M_UR50D"
    ):
        super().__init__()
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        self.encoder = AutoModelForMaskedLM.from_pretrained(base_model)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch):
        x1, x2, y, atm1, atm2 = batch
        z1 = self._pool(self.encoder(x1, output_hidden_states=True).hidden_states, atm1)
        z2 = self._pool(self.encoder(x2, output_hidden_states=True).hidden_states, atm2)

        z = torch.cat((z1, z2), 1)
        logits = self.head(z)
        # softmax
        y_hat = torch.softmax(logits, dim=1)
        return y_hat

    def training_step(self, batch, batch_idx):
        y = batch[2]
        y_hat = self.forward(batch)

        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[2]
        y_hat = self.forward(batch)
        loss = nn.functional.cross_entropy(y_hat, y)

        self.valid_acc(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.log("valid_acc", self.valid_acc, on_epoch=True, prog_bar=True)

    def predict(self, seq1: str, seq2: str, tokenizer: transformers.AutoTokenizer):
        """Predicts the probability of a protein-protein interaction
        Args:
            seq1 (str): Sequence of the first protein as a string
            seq2 (str): Sequence of the second protein as a string
            tokenizer (transformers.AutoTokenizer): Tokenizer for the model
        Returns:
            y_hat (torch.Tensor): Probability of a protein-protein interaction
        """
        ts_1 = tokenizer(
            [seq1],
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
        )
        ts_2 = tokenizer(
            [seq2],
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
        )

        x1 = ts_1["input_ids"]
        x2 = ts_2["input_ids"]
        atm1 = ts_1["attention_mask"]
        atm2 = ts_2["attention_mask"]

        return self.forward((x1, x2, None, atm1, atm2))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        Pool (average) the output of the last layer of a transformer model fitering out padding tokens.

        Args:
            hidden_states : The output of the last layer of a transformer model,
                with shape (batch_size, sequence_length, hidden_size).
            attention_mask : An attention mask tensor with shape (batch_size, sequence_length),
                where 1 indicates an input token and 0 indicates a padding token.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) containing the pooled representations
            of the input sequence.

        """
        last_layer = hidden_states[-1]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_layer.size()).float()
        )
        return torch.sum(last_layer * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
