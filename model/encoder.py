import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        """
        input_seq: [batch_size, seq_len]
        Returns:
          outputs: [batch_size, seq_len, hidden_dim * 2] (bidirectional)
          hidden: [num_layers * 2, batch_size, hidden_dim]
        """
        embedded = self.dropout(self.embedding(input_seq))  # [batch, seq_len, emb_dim]
        outputs, hidden = self.gru(embedded)                 # outputs: [batch, seq_len, 2*hidden]
        return outputs, hidden
