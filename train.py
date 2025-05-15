from preprocess.squad_loader import load_squad_dataset, preprocess_samples
from dataset.qg_dataset import QuestionGenerationDataset, collate_fn
from torch.utils.data import DataLoader

# 1. Carica i dati grezzi
samples = load_squad_dataset("data/squad_train.json")

# 2. Preprocessali
encoded_samples, vocab = preprocess_samples(samples[:1000])  # subset per test

# 3. Crea il Dataset
dataset = QuestionGenerationDataset(encoded_samples)

# 4. Crea il DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 5. Test
for batch in dataloader:
    context, answer, question = batch
    print("Context shape:", context.shape)
    print("Answer shape:", answer.shape)
    print("Question shape:", question.shape)
    break
