import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch, hidden]
        encoder_outputs: [batch, seq_len, hidden*2]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Ripeti hidden per concatenare con ogni token dell'encoder
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, seq_len, hidden]
        scores = self.v(energy).squeeze(2)  # [batch, seq_len]
        attn_weights = F.softmax(scores, dim=1)  # [batch, seq_len]

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden*2]
        context = context.squeeze(1)  # [batch, hidden*2]
        return context, attn_weights

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = Attention(dec_hidden_dim)
        self.gru = nn.GRU(emb_dim + enc_hidden_dim * 2, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim * 2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs):
        """
        input_token: [batch]         (indice parola precedente)
        hidden: [1, batch, hidden]   (stato decoder corrente)
        encoder_outputs: [batch, seq_len, enc_hidden*2]
        """
        input_token = input_token.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, emb_dim]

        # Attenzione
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch, enc_hidden*2]

        # Input GRU: concateni embedded + contesto
        gru_input = torch.cat((embedded.squeeze(1), context), dim=1).unsqueeze(1)  # [batch, 1, emb+enc*2]
        output, hidden = self.gru(gru_input, hidden)  # output: [batch, 1, dec_hidden]

        output = output.squeeze(1)  # [batch, dec_hidden]
        combined = torch.cat((output, context, embedded.squeeze(1)), dim=1)  # [batch, dec+enc*2+emb]
        prediction = self.fc_out(combined)  # [batch, vocab_size]

        return prediction, hidden, attn_weights
