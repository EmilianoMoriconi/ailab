import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
from datetime import datetime

def train_model(encoder, decoder, dataloader, vocab, device, num_epochs=5, teacher_forcing_ratio=0.5):
    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    history = []

    os.makedirs("saved", exist_ok=True)
    
    best_loss = float('inf')
    
    log_data = {
        "config": {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_epochs": num_epochs,
            "batch_size": dataloader.batch_size,
            "learning_rate": 0.001,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "device": str(device),
            "emb_dim": encoder.embedding.embedding_dim,
            "hidden_dim": encoder.gru.hidden_size
        },
        "last": {},
        "best": {}
    }
    
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        epoch_loss = 0
        total_tokens = 0
        correct_tokens = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for context, _, question in pbar:
            context = context.to(device)
            question = question.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(context)
            decoder_hidden = encoder_hidden[0:1] + encoder_hidden[1:2]  # somma direzioni

            decoder_input = question[:, 0]  # <sos>
            max_len = question.size(1)
            loss = 0

            for t in range(1, max_len):
                output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                target = question[:, t]

                loss += criterion(output, target)

                # Token-level accuracy
                pred = output.argmax(1)
                mask = (target != 0)
                correct_tokens += (pred == target).masked_select(mask).sum().item()
                total_tokens += mask.sum().item()

                # Teacher forcing
                decoder_input = target if torch.rand(1).item() < teacher_forcing_ratio else pred

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            batch_loss = loss.item() / (max_len - 1)
            epoch_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Token Accuracy: {accuracy:.4f}")

        # Salva sempre l'ultimo modello
        torch.save(encoder.state_dict(), "saved/encoder_last.pt")
        torch.save(decoder.state_dict(), "saved/decoder_last.pt")
        
        # Log "last"
        log_data["last"] = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "token_accuracy": accuracy
        }

        # Salva il migliore (in base alla loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), "saved/encoder_best.pt")
            torch.save(decoder.state_dict(), "saved/decoder_best.pt")
            log_data["best"] = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "token_accuracy": accuracy
            }
            print(f"✔️  Miglior modello salvato (loss = {best_loss:.4f})")


    # Aggiunge la run corrente a una lista cumulativa
    log_file = "saved/training_log.json"

    # Se esiste già un file log, lo carica
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            all_logs = json.load(f)
    else:
        all_logs = []

    # Aggiunge la nuova run alla lista
    all_logs.append(log_data)

    # Scrive l'intera lista nel file
    with open(log_file, "w") as f:
        json.dump(all_logs, f, indent=2)
