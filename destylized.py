import json
import nltk
import time
from tqdm import tqdm
import google_translator

EXCLUDE_SET = {'~', '`', '@', '#', '$', '%', '^', '&', '*', '(', ')',
               '-', '+', '=', '[', ']', '{', '}', '|', ':', ';', '\'',
               '"', ',', '.', '<', '>', '/', '?', '!', '...'}

# args
DATA_PATH = './dataset/bigbang.json'
SAVE_PATH = './dataset/destylized_bigbang.json'
TMP = './dataset/destylized.tmp.txt'
MAX_LINE = 32


example_list = []
destylized_sentences_list = []
with open(DATA_PATH, 'r') as src:
    with open(TMP, 'w') as tmp:
        lines = src.readlines()
        num_lines = len(lines)
        start = 0
        end = min(start + MAX_LINE, num_lines)
        for _ in tqdm(range(num_lines // MAX_LINE + 1)):
            response_list = []
            for line in lines[start:end]:
                example = json.loads(line)
                example_list.append(example)
                response_list.append(example['response'])
            destylized_sentences = google_translator.transform('\n'.join(response_list)).strip()
            time.sleep(5)
            for sentence in destylized_sentences.split('\n'):
                sentence_tokens = nltk.tokenize.word_tokenize(sentence)
                filter_tokens = []
                for token in sentence_tokens:
                    if token == 'num':
                        filter_tokens.append('<num>')
                    elif token in EXCLUDE_SET:
                        continue
                    else:
                        filter_tokens.append(token)
                sentence = ' '.join(filter_tokens)
                tmp.write(sentence + '\n')
                destylized_sentences_list.append(sentence)
            start = end
            end = min(start + MAX_LINE, num_lines)

with open(SAVE_PATH, 'w') as tgt:
    for i, example in enumerate(example_list):
        example['destylized'] = destylized_sentences_list[i]
        tgt.write(json.dumps(example, ensure_ascii=False) + '\n')
