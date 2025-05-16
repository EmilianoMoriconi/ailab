from preprocess.squad_loader import load_squad_dataset, preprocess_samples
from dataset.qg_dataset import QuestionGenerationDataset, collate_fn
from model.encoder import EncoderGRU
from model.decoder import DecoderGRU
from train_loop import train_model
from torch.utils.data import DataLoader
import torch

# 1. Carica e preprocessa i dati
samples = load_squad_dataset("question_gen/data/squad_train.json")
samples = samples[:8000]  
encoded_samples, vocab = preprocess_samples(samples)

# 2. Crea dataset e dataloader
dataset = QuestionGenerationDataset(encoded_samples)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 3. Imposta i parametri
vocab_size = len(vocab)
emb_dim = 128
hidden_dim = 256

# 4. Inizializza modelli
encoder = EncoderGRU(vocab_size, emb_dim, hidden_dim)
decoder = DecoderGRU(vocab_size, emb_dim, hidden_dim, hidden_dim)

# ⬇️ Carica pesi se presenti
import os
if os.path.exists("question_gen/saved/encoder_last.pt") and os.path.exists("question_gen/saved/decoder_last.pt"):
    encoder.load_state_dict(torch.load("question_gen/saved/encoder_last.pt"))
    decoder.load_state_dict(torch.load("question_gen/saved/decoder_last.pt"))
    print("✅ Modelli ripristinati")
else:
    print("ℹ️ Nessun modello salvato, partenza da zero")


# 5. Allenamento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(encoder, decoder, dataloader, vocab, device, num_epochs=20)

torch.save(encoder.state_dict(), "question_gen/saved/encoder.pt")
torch.save(decoder.state_dict(), "question_gen/saved/decoder.pt")

