from preprocess.squad_loader import load_squad_dataset, preprocess_samples
from dataset.qg_dataset import QuestionGenerationDataset, collate_fn
from model.encoder import EncoderGRU
from model.decoder import DecoderGRU
from train_loop import train_model
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
torch.cuda.empty_cache()

# 1. Carica e preprocessa i dati
train_samples = load_squad_dataset("data/SQuAD_it-train.json")
train_samples = train_samples[:10000]
test_samples = load_squad_dataset("data/SQuAD_it-test.json")
test_samples = test_samples[:10000]

# 🔄 Carica vocabolario se esiste, altrimenti crealo e salvalo
if os.path.exists("saved/vocab.pkl"):
    with open("saved/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    encoded_samples, _ = preprocess_samples(train_samples, vocab=vocab, build_vocab=False)
    print("✅ Vocabolario caricato da file.")
else:
    encoded_samples, vocab = preprocess_samples(train_samples)
    with open("saved/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("💾 Vocabolario salvato in 'saved/vocab.pkl'")

preprocess_test_samples, _ = preprocess_samples(test_samples, vocab=vocab, build_vocab=False)

train_dataset = QuestionGenerationDataset(encoded_samples)
val_dataset = QuestionGenerationDataset(preprocess_test_samples)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 3. Imposta i parametri
vocab_size = len(vocab)
emb_dim = 128
hidden_dim = 256

# 4. Inizializza modelli
encoder = EncoderGRU(vocab_size, emb_dim, hidden_dim)
decoder = DecoderGRU(vocab_size, emb_dim, hidden_dim, hidden_dim)

# ⬇️ Carica pesi se presenti
if os.path.exists("saved/encoder_last.pt") and os.path.exists("saved/decoder_last.pt"):
    encoder.load_state_dict(torch.load("saved/encoder_last.pt"))
    decoder.load_state_dict(torch.load("saved/decoder_last.pt"))
    print("✅ Modelli ripristinati")
else:
    print("ℹ️ Nessun modello salvato, partenza da zero")


# 5. Allenamento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(encoder, decoder, train_dataloader, val_dataloader, vocab, device, num_epochs=1)


torch.save(encoder.state_dict(), "saved/encoder.pt")
torch.save(decoder.state_dict(), "saved/decoder.pt")

