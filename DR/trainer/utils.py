import os
import csv
import torch
import logging
from rouge import Rouge
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, recall_score, precision_score


logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.args = args
        self.step = 0
        self.rouge = Rouge()
        self.rouge_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-l': {'r': 0, 'p': 0, 'f': 0}}

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def cal_performance(self, yhat, y, search_idx, hypo):

        with torch.no_grad():
            y = y.squeeze(1)
            yhat = yhat.squeeze(1).max(dim=-1)[1]

            acc = (yhat == y).float().mean()
            f1 = f1_score(y.cpu(), yhat.cpu(), average='macro')

            y = y.cpu().numpy().tolist()
            yhat = yhat.cpu().numpy().tolist()

            search_idx = search_idx.squeeze(1).cpu().numpy().tolist()

            assert len(y) == len(yhat) == len(search_idx)

            for idx in range(len(y)):
                checker = (search_idx[idx], yhat[idx], y[idx])
                hypo.append(checker)

        return acc, f1

    def cal_rouge(self, checker, memory_db):
        path_to_test_data = self.args.path_to_data + '/' + self.args.test_data
        response_set = []
        with open(path_to_test_data, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')

            for line in lines:
                response = line[1].strip()
                response_set.append(response)

        for idx in range(len(checker)):
            search_idx, yhat, _ = checker[idx]
            gold_response = response_set[idx]

            hypo = memory_db[search_idx][yhat].split('[SEP]')[1].strip()
            score = self.rouge.get_scores(hypo, gold_response)[0]

            for metric, scores in self.rouge_scores.items():
                for key, value in scores.items():
                    self.rouge_scores[metric][key] += score[metric][key]

            self.step += 1

    def avg_rouge(self):
        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] /= self.step

        return self.rouge_scores

    def performance_check(self, cp):
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Valid Loss: {cp["vl"]:.4f}==')
        print(f'\t==Train Acc: {cp["ta"]:.4f} | Valid Acc: {cp["va"]:.4f}==')
        print(f'\t==Train F1: {cp["tf"]:.4f} | Valid F1: {cp["vf"]:.4f}==\n')

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def save_config(self, cp):
        config = "Config>>\n"
        for idx, (key, value) in enumerate(self.args.__dict__.items()):
            cur_kv = str(key) + ': ' + str(value) + '\n'
            config += cur_kv
        config += 'Epoch: ' + str(cp["ep"]) + '\t' + 'Valid loss: ' + str(cp['vl']) + '\n'

        with open(self.args.path_to_save+'config.txt', "w") as f:
            f.write(config)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + config['args'].ckpt

        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']
            torch.save(config['model'].state_dict(), sorted_path)
            self.save_config(cp)
            print(f'\n\t## SAVE Valid Loss: {cp["vl"]:.4f} ##')

        else:
            pco['early_stop_patient'] += 1
            if pco['early_stop_patient'] == config['args'].patient:
                pco['early_stop'] = True

        self.performance_check(cp)