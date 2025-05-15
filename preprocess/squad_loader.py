import nltk
import json
from nltk.tokenize import word_tokenize
from collections import Counter
from question_gen.utils.vocab import Vocab


nltk.download('punkt')

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





def tokenize(text):
    return word_tokenize(text.lower())

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
        qst_ids = [vocab.word2idx['<sos>']] + vocab.encode(qst) + [vocab.word2idx['<eos>']]
        encoded_samples.append({
            'context_ids': ctx_ids,
            'answer_ids': ans_ids,
            'question_ids': qst_ids
        })

    return encoded_samples, vocab
