import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class QuestionGenerationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item['context_ids'], dtype=torch.long),
            torch.tensor(item['answer_ids'], dtype=torch.long),
            torch.tensor(item['question_ids'], dtype=torch.long)
        )

# Funzione per il padding e il batching
def collate_fn(batch):
    context_batch, answer_batch, question_batch = zip(*batch)

    context_padded = pad_sequence(context_batch, batch_first=True, padding_value=0)
    answer_padded  = pad_sequence(answer_batch, batch_first=True, padding_value=0)
    question_padded = pad_sequence(question_batch, batch_first=True, padding_value=0)

    return context_padded, answer_padded, question_padded

