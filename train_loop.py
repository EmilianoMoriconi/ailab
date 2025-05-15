import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(encoder, decoder, dataloader, vocab, device, num_epochs=5, teacher_forcing_ratio=0.5):
    # Sposta i modelli su GPU o CPU
    encoder.to(device)
    decoder.to(device)

    # Funzione di perdita e ottimizzatori
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignora il padding
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            context, _, question = batch
            context = context.to(device)
            question = question.to(device)

            # Reset gradienti
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # 1. ENCODER
            encoder_outputs, encoder_hidden = encoder(context)

            # Usa la somma dei due hidden state (forward + backward) per inizializzare il decoder
            decoder_hidden = encoder_hidden[0:1] + encoder_hidden[1:2]

            # 2. DECODER: inizializza input con <sos>
            decoder_input = question[:, 0]  # primo token di ogni sequenza (sos)
            max_len = question.size(1)
            loss = 0

            for t in range(1, max_len):
                # output: [batch, vocab_size]
                output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)

                # Target corrente: parola successiva nella sequenza
                target = question[:, t]
                loss += criterion(output, target)

                # Teacher forcing: usa parola vera come input al prossimo step
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                top1 = output.argmax(1)
                decoder_input = target if use_teacher else top1

            # Backprop e ottimizzazione
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            batch_loss = loss.item() / (max_len - 1)
            epoch_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
