import nltk
import json
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from utils.vocab import Vocab



def load_squad_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context'].strip()

            for qa in paragraph['qas']:
                question = qa['question'].strip()

                for answer in qa['answers']:
                    ans_text = answer['text'].strip()
                    samples.append({
                        'context': context,
                        'answer': ans_text,
                        'question': question
                    })

    return samples




tokenizer = TreebankWordTokenizer()

def tokenize(text):
    return tokenizer.tokenize(text.lower())

def preprocess_samples(samples, vocab=None, build_vocab=True):
    contexts_tok = [tokenize(s['context']) for s in samples]
    questions_tok = [tokenize(s['question']) for s in samples]
    answers_tok = [tokenize(s['answer']) for s in samples]

    if build_vocab:
        vocab = Vocab(min_freq=2)
        vocab.build_vocab(contexts_tok + questions_tok + answers_tok)

    encoded_samples = []
    for ctx, qst, ans in zip(contexts_tok, questions_tok, answers_tok):
        ctx_ids = vocab.encode(ctx)
        ans_ids = vocab.encode(ans)
        qst_ids = [vocab.stoi['<sos>']] + vocab.encode(qst) + [vocab.stoi['<eos>']]
        encoded_samples.append({
            'context_ids': ctx_ids,
            'answer_ids': ans_ids,
            'question_ids': qst_ids
        })

    return encoded_samples, vocab
