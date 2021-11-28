import torch
import torch.nn as nn
from einops import repeat
from data.dataloader import get_cur_db_data, get_top_db_data
from transformers import BertModel, BertConfig


class ScoreModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(ScoreModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(p=args.dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.cos = nn.CosineSimilarity(dim=-1)

        self.bert_config = BertConfig()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.ws_parm = nn.Parameter(torch.zeros(1, self.bert_config.hidden_size))

        self.bert.projection = nn.Linear(self.bert_config.hidden_size * 2, 1)
        self.wb = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.wc = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)

    def forward(self, inputs, memory_db, mode):
        if mode != 'test':
            db_inputs = get_cur_db_data(inputs['search_idx'], memory_db)

            post_hidden1, _ = self.bert(input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'],
                                        token_type_ids=inputs['token_type_ids'],
                                        return_dict=False)

            db_response_tensor = torch.ones(post_hidden1.size(0), 1, self.bert_config.hidden_size)

            fuse_param = repeat(self.ws_parm, '() h -> b h', b=post_hidden1.size(0))

            post1 = self.wb(post_hidden1[:, :1].squeeze(1))
            prior1 = self.wb(fuse_param)

            for idx in range(self.args.top_k):
                cur_db_inputs = get_top_db_data(self.args, self.tokenizer, db_inputs, idx)
                post_hidden2, _ = self.bert(input_ids=cur_db_inputs['input_ids'],
                                            attention_mask=cur_db_inputs['attention_mask'],
                                            token_type_ids=cur_db_inputs['token_type_ids'],
                                            return_dict=False)

                db_response = post_hidden2[:, :1]
                db_response_tensor = torch.cat([db_response_tensor, db_response.cpu()], dim=1)

            db_response_tensor = self.wc(db_response_tensor[:, 1:, :].to(self.args.device)).transpose(-1, -2)

            post_logits = torch.matmul(post1.unsqueeze(1).to(self.args.device),
                                       db_response_tensor)

            prior_logits = torch.matmul(prior1.unsqueeze(1).to(self.args.device),
                                        db_response_tensor)

            return post_logits.squeeze(1), prior_logits.squeeze(1)

        else:
            db_inputs = get_cur_db_data(inputs['search_idx'], memory_db)
            fuse_param = repeat(self.ws_parm, '() h -> b h', b=len(db_inputs))

            prior1 = self.wb(fuse_param)
            db_response_tensor = torch.ones(prior1.size(0), 1, self.bert_config.hidden_size)

            for idx in range(self.args.top_k):
                cur_db_inputs = get_top_db_data(self.args, self.tokenizer, db_inputs, idx)
                post_hidden2, _ = self.bert(input_ids=cur_db_inputs['input_ids'],
                                            attention_mask=cur_db_inputs['attention_mask'],
                                            token_type_ids=cur_db_inputs['token_type_ids'],
                                            return_dict=False)

                db_response = post_hidden2[:, :1]
                db_response_tensor = torch.cat([db_response_tensor, db_response.cpu()], dim=1)

            db_response_tensor = self.wc(db_response_tensor[:, 1:, :].to(self.args.device)).transpose(-1, -2)
            prior_logits = torch.matmul(prior1.unsqueeze(1).to(self.args.device),
                                        db_response_tensor)

            return _, prior_logits.squeeze(1)