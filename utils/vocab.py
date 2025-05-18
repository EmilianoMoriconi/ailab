from collections import Counter


class Vocab:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.stoi = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.itos = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.word_freq = Counter()

    def build_vocab(self, token_lists):
        for tokens in token_lists:
            self.word_freq.update(tokens)

        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def encode(self, tokens):
        return [self.stoi.get(t, self.stoi['<unk>']) for t in tokens]

    def decode(self, indices):
        return [self.itos[i] for i in indices]

    def __len__(self):
        return len(self.stoi)
    
    def get_itos(self):
        """Restituisce la lista di parole (index-to-string)."""
        return self.itos
