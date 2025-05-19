import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import time
import os
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def train_model(encoder, decoder, dataloader, val_loader, vocab, device, num_epochs=5, teacher_forcing_ratio=0.5):
    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)



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
    
    start_time = time.time()

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

        # 🔍 Validazione con BLEU
        references, candidates = evaluate_on_validation(encoder, decoder, val_loader, vocab, device)
        bleu = compute_bleu(references, candidates)
        print(f"🔵 BLEU score: {bleu:.4f}")

        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device)
            max_allocated = torch.cuda.max_memory_allocated(device)

            allocated_mb = allocated / 1024 ** 2
            max_allocated_mb = max_allocated / 1024 ** 2

            allocated_gb = allocated / 1024 ** 3
            max_allocated_gb = max_allocated / 1024 ** 3

            print(f"📈 VRAM attualmente allocata: {allocated_mb:.2f} MB ({allocated_gb:.2f} GB)")
            print(f"📉 Picco massimo VRAM usata in epoca: {max_allocated_mb:.2f} MB ({max_allocated_gb:.2f} GB)")

            # resetta il contatore per l’epoca successiva
            torch.cuda.reset_max_memory_allocated(device)


        examples_to_save = []
        for i in range(min(3, len(references))):
            ref, ctx = references[i]
            examples_to_save.append({
                "context": ctx,
                "reference": ref,
                "generated": candidates[i]
            })

        epoch_examples = {
            "epoch": epoch + 1,
            "examples": examples_to_save
        }

        examples_log_file = "saved/examples_log.json"

        # Se il file esiste, carica e aggiorna
        if os.path.exists(examples_log_file):
            with open(examples_log_file, "r") as f:
                all_examples = json.load(f)
        else:
            all_examples = []

        all_examples.append(epoch_examples)

        with open(examples_log_file, "w") as f:
            json.dump(all_examples, f, indent=2)

        # Salva sempre l'ultimo modello
        torch.save(encoder.state_dict(), "saved/encoder_last.pt")
        torch.save(decoder.state_dict(), "saved/decoder_last.pt")
        
        # Log "last"
        log_data["last"] = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "token_accuracy": accuracy,
            "bleu": bleu
        }

        # Salva il migliore (in base alla loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), "saved/encoder_best.pt")
            torch.save(decoder.state_dict(), "saved/decoder_best.pt")
            log_data["best"] = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "token_accuracy": accuracy,
                "bleu": bleu
            }
            print(f"✔️  Miglior modello salvato (loss = {best_loss:.4f})")

    end_time = time.time()
    elapsed = end_time - start_time

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"⏱️  Training completato in {minutes} min {seconds} sec.")

    
    
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

def tensor_to_text(tensor, vocab):
    """Converte un tensore di token in una stringa (esclude padding e <sos>/<eos>)."""
    itos = vocab.get_itos()
    words = []
    for token in tensor:
        if token.item() == vocab.stoi["<eos>"]:
            break
        if token.item() != 0 and token.item() != vocab.stoi["<sos>"]:
            words.append(itos[token.item()])
    return " ".join(words)

def evaluate_on_validation(encoder, decoder, val_loader, vocab, device, max_len=30):
    encoder.eval()
    decoder.eval()
    references = []
    candidates = []

    with torch.no_grad():
        for context, _, question in val_loader:
            context = context.to(device)
            encoder_outputs, encoder_hidden = encoder(context)
            decoder_hidden = encoder_hidden[0:1] + encoder_hidden[1:2]
            decoder_input = torch.tensor([vocab.stoi["<sos>"]] * context.size(0), device=device)

            outputs = []
            for _ in range(max_len):
                output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                top1 = output.argmax(1)
                outputs.append(top1.unsqueeze(1))
                decoder_input = top1

            outputs = torch.cat(outputs, dim=1)
            for i in range(outputs.size(0)):
                ref = tensor_to_text(question[i], vocab)
                hyp = tensor_to_text(outputs[i], vocab)
                ctx = tensor_to_text(context[i], vocab)

                references.append((ref, ctx))      # salva reference con relativo context
                candidates.append(hyp)


    return references, candidates

def compute_bleu(references, candidates):
    scores = []
    for ref, cand in zip(references, candidates):
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        # score = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5))  # BLEU-2
        score = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0))  # BLEU-1
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0