import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TokenClassificationPipeline
from pathlib import Path

# === Carica modello e tokenizer salvati ===
model_path = "./model"
assert Path(model_path).exists(), "❌ Cartella ./model non trovata. Hai completato il training?"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# === Usa la pipeline HuggingFace ===
pipe = TokenClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",  # aggrega B/I token in una singola entità
    device=0 if torch.cuda.is_available() else -1
)

# === Input da terminale ===
print("🔎 Inserisci una frase in italiano per il riconoscimento NER (CTRL+C per uscire)")
while True:
    try:
        text = input("\n📝 Frase: ")
        preds = pipe(text)

        print("\n📌 Entità riconosciute:")
        if not preds:
            print(" (nessuna)")
        for ent in preds:
            print(f" 🟢 {ent['word']} → {ent['entity_group']} ({ent['score']:.2f})")

    except KeyboardInterrupt:
        print("\n👋 Fine.")
        break
