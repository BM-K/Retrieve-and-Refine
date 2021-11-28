import csv
import torch
import logging
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer):
        self.args = args
        self.metric = metric

        self.tokenizer = tokenizer
        self.file_path = file_path

        self.labels = []
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []
        self.search_idx = []

        """
        BERT
        [CLS] 101
        [PAD] 0
        [UNK] 100
        
        Roberta
        init token, idx = <s>, 0
        pad token, idx = <pad>, 1
        unk token, idx = <unk>, 3
        eos token, idx = </s>, 2
        """

        self.init_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)

    def load_data(self, type):

        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')

            for line in lines:
                self.data2tensor(line)

        assert len(self.input_ids) == \
               len(self.attention_mask) == \
               len(self.labels)

    def data2tensor(self, line):

        question, response, search_idx, label = line[0].strip(), line[1].strip(), line[2].strip(), line[3].strip()

        sentence_tokens = self.tokenizer(question,
                                         response,
                                         truncation=True,
                                         return_tensors="pt",
                                         max_length=self.args.max_len,
                                         pad_to_max_length="right")

        self.input_ids.append(sentence_tokens['input_ids'].squeeze(0))
        self.attention_mask.append(sentence_tokens['attention_mask'].squeeze(0))
        self.token_type_ids.append(sentence_tokens['token_type_ids'].squeeze(0))
        self.search_idx.append(int(search_idx))
        self.labels.append(int(label))

        return True

    def __getitem__(self, index):

        input_data = {'input_ids': self.input_ids[index].to(self.args.device),
                      'attention_mask': self.attention_mask[index].to(self.args.device),
                      'token_type_ids': self.token_type_ids[index].to(self.args.device),
                      'search_idx': torch.IntTensor([self.search_idx[index]]).to(self.args.device),
                      'labels': torch.LongTensor([self.labels[index]]).to(self.args.device)}

        return input_data

    def __len__(self):
        return len(self.labels)


def get_loader(args, metric):
    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.test_data

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer)
        valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer)
        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer)
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=False)}

    else:
        logger.info("Error: None type loader")
        exit()

    return loader, tokenizer


def memory_db(args):
    path_to_db_data = args.path_to_data + '/' + args.db_path
    db_memory = []
    with open(path_to_db_data, "r", encoding="utf-8") as file:
        lines = csv.reader(file, delimiter="\t", quotechar='"')

        for line in lines:
            response = line[:args.top_k]
            query = line[-1]
            cur_sentences = [query + "[SEP] " + r for r in response]

            db_memory.append(cur_sentences)

    print(f"\nDB LENGTH: {len(db_memory)}\n")

    return db_memory


def get_cur_db_data(search_idx, db_data):
    cur_batch_db_data = []
    search_idx = search_idx.cpu().numpy().tolist()

    for idx in search_idx:
        cur_db_data = db_data[idx[0]]
        cur_batch_db_data.append(cur_db_data)

    return cur_batch_db_data


def get_top_db_data(args, tokenizer, db_data, idx):
    cur_idx_db_data = []
    for i in range(len(db_data)):
        cur_idx_db_data.append(db_data[i][idx])

    tokens = tokenizer(cur_idx_db_data,
                       truncation=True,
                       return_tensors="pt",
                       max_length=args.max_len,
                       pad_to_max_length="right")

    return tokens.to(args.device)


if __name__ == '__main__':
    get_loader('test')