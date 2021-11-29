import csv
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util

seed = 1234

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model.to(torch.device('cuda:1'))

total_data = []
reddit_data_path = "reddit_data.tsv"
with open(reddit_data_path, "r", encoding="utf-8") as file:
    lines = csv.reader(file, delimiter="\t", quotechar='"')
    for line in lines:
        query, response = line
        total_data.append((query, response))

print(f"Total data length: {len(total_data)}")
random.shuffle(total_data)

train_data_path = 'train_set.tsv'
corpus_data_path = 'database_set.tsv'
db_data = total_data[:1000000]
train_data = total_data[1000000:1010000]
print(f"DB data: {len(db_data)}\tTrain data: {len(train_data)}")

with open(train_data_path, "w", encoding="utf-8") as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    for line in train_data:
        query, response = line
        tsv_writer.writerow([query, response])

with open(corpus_data_path, "w", encoding="utf-8") as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    for line in db_data:
        query, response = line
        tsv_writer.writerow([query, response])

queries = []
corpus = []
with open(train_data_path, "r", encoding="utf-8") as file:
    lines = csv.reader(file, delimiter="\t", quotechar='"')
    for line in lines:
        query, response = line
        queries.append(query + " [SEP] " + response)

with open(corpus_data_path, "r", encoding="utf-8") as file:
    lines = csv.reader(file, delimiter="\t", quotechar='"')
    for line in lines:
        query_prime, response_prime = line
        # corpus.append(response_prime)
        corpus.append(query_prime + " [SEP] " + response_prime)

top_k = 10
hard_label = 1
batch_size = 512
length_of_corpus = len(corpus)

database = []
search_idx = 0
total_data = []

print(f"Start DB Embedding \n")
tensor_stack = torch.ones(1, 768)
for start in range(0, len(corpus), batch_size):
    corpus_tokens = corpus[start : start + batch_size]

    with torch.no_grad():
        corpus_tokens = tokenizer(corpus_tokens, padding=True, truncation=True, return_tensors="pt")
        corpus_tokens.to(torch.device('cuda:1'))
        corpus_embeddings = model(**corpus_tokens, output_hidden_states=True, return_dict=True).pooler_output

    tensor_stack = torch.cat([tensor_stack, corpus_embeddings.cpu()], dim=0)

tensor_stack = tensor_stack[1:]
print(f"Total DB embedding length: {len(tensor_stack)}")

for query in tqdm(queries):
    with torch.no_grad():
        query_tokens = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        query_tokens.to(torch.device('cuda:1'))
        query_embedding = model(**query_tokens, output_hidden_states=True, return_dict=True).pooler_output

    q, r = query.split('[SEP]')

    """
    tensor_stack = torch.ones(1, query_embedding.size(-1))

    for start in range(0, len(corpus), batch_size):
        current_corpus = corpus[start : start + batch_size]
        corpus_tokens = [q + "[SEP] " + c for c in current_corpus]

        with torch.no_grad():
            corpus_tokens = tokenizer(corpus_tokens, padding=True, truncation=True, return_tensors="pt")
            corpus_tokens.to(torch.device('cuda'))
            corpus_embeddings = model(**corpus_tokens, output_hidden_states=True, return_dict=True).pooler_output

        tensor_stack = torch.cat([tensor_stack, corpus_embeddings.cpu()], dim=0)

    tensor_stack = tensor_stack[1:]
    """

    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), tensor_stack)[0]
    cos_scores = cos_scores.cpu()

    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    top_k_setting = []
    top_label = 1

    for idx in top_results[0:hard_label]:
        top_k_setting.append((corpus[idx].strip(), top_label))
        top_label += 1

    random_list = [random.randint(0, len(tensor_stack)-hard_label) for r in range(top_k-hard_label)]
    while(idx in random_list):
        random_list = [random.randint(0, len(tensor_stack)-hard_label) for r in range(top_k - hard_label)]

    for step, idx in enumerate(random_list):
        top_k_setting.append((corpus[idx].strip(), top_label))
        top_label += 1

    random.shuffle(top_k_setting)
    best_score_index = [idx for idx, (response, score) in enumerate(top_k_setting) if score == 1]
    top_k_setting.append(search_idx)
    top_k_setting.append(q)

    train_data = (q, r, search_idx, best_score_index[0])
    database.append(top_k_setting)
    total_data.append(train_data)
    search_idx += 1

train_data_path = 'train_dataset.tsv'
corpus_data_path = 'database_set.tsv'

with open(train_data_path, "w", encoding="utf-8") as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    for line in total_data:
        query, response, search_idx, best_score_index = line
        tsv_writer.writerow([query, response, search_idx, best_score_index])

with open(corpus_data_path, "w", encoding="utf-8") as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    for line in database:
        current_response = line[:top_k]
        search_idx, query = line[top_k:]

        current_response = [r for r, score in current_response]
        current_response.append(search_idx)
        current_response.append(query)

        tsv_writer.writerow(current_response)
