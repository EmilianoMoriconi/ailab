import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os

def train_model(encoder, decoder, dataloader, vocab, device, num_epochs=5, teacher_forcing_ratio=0.5):
    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    history = []

    os.makedirs("saved", exist_ok=True)

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

        # Salvataggio modelli per epoca
        torch.save(encoder.state_dict(), f"saved/encoder_epoch{epoch+1}.pt")
        torch.save(decoder.state_dict(), f"saved/decoder_epoch{epoch+1}.pt")

        # Tracking log
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "token_accuracy": accuracy
        })

        with open("saved/training_log.json", "w") as f:
            json.dump(history, f, indent=2)
