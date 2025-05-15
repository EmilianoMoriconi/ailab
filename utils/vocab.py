from collections import Counter


class Vocab:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.word_freq = Counter()

    def build_vocab(self, token_lists):
        for tokens in token_lists:
            self.word_freq.update(tokens)

        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def encode(self, tokens):
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]

    def decode(self, indices):
        return [self.idx2word[i] for i in indices]

    def __len__(self):
        return len(self.word2idx)
