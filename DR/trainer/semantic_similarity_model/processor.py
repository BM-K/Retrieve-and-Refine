import logging
from apex import amp
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim
from trainer.loss import Loss
from trainer.utils import Metric
from transformers import get_linear_schedule_with_warmup
from trainer.semantic_similarity_model.models import ScoreModel
from data.dataloader import get_loader, memory_db, get_cur_db_data

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.loss_fn = Loss(args)
        self.metric = Metric(args)
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': 0, 'iter': 0, 'acc': 0, 'f1': 0}
        self.sorted_path = args.path_to_save + args.ckpt
        self.memory_db = memory_db(self.args)
        self.hypo = []

    def run(self, inputs, mode='Normal'):
        post_logits, prior_logits = self.config['model'](inputs, self.memory_db, mode)

        if mode == 'Normal':
            post_loss = self.loss_fn.base(self.config, post_logits, inputs['labels'])
            prior_loss = self.loss_fn.base(self.config, prior_logits, inputs['labels'])
            kd_loss = self.loss_fn.kd_loss(self.config, post_logits, prior_logits)

            loss = post_loss + prior_loss + kd_loss
        else:
            loss = self.loss_fn.base(self.config, prior_logits, inputs['labels'])

        acc, f1 = self.metric.cal_performance(prior_logits,
                                              inputs['labels'],
                                              inputs['search_idx'],
                                              self.hypo)

        return loss, acc, f1

    def progress(self, loss, acc, f1):
        self.model_progress['iter'] += 1
        self.model_progress['loss'] += loss
        self.model_progress['acc'] += acc
        self.model_progress['f1'] += f1

    def return_value(self, mode='Normal'):
        loss = self.model_progress['loss'].cpu().numpy() / self.model_progress['iter']
        acc = self.model_progress['acc'] / self.model_progress['iter']
        f1 = self.model_progress['f1'] / self.model_progress['iter']

        if mode == 'Normal':
            return loss, acc, f1, 0
        else:
            return loss, acc, f1, self.metric.avg_rouge()

    def get_object(self, tokenizer, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=self.args.warmup_ratio * train_total,
            num_training_steps=train_total)

        return scheduler

    def model_setting(self):
        loader, tokenizer = get_loader(self.args, self.metric)

        model = ScoreModel(self.args, tokenizer)
        model.to(self.args.device)

        criterion, optimizer = self.get_object(tokenizer, model)

        if self.args.test == 'False':
            scheduler = self.get_scheduler(optimizer, loader['train'])
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'args': self.args,
                  'model': model}

        if config['args'].fp16 == 'True' and config['args'].test == 'False':
            config['model'], config['optimizer'] = amp.initialize(
                config['model'], config['optimizer'], opt_level=config['args'].opt_level)

        self.config = config

        return self.config

    def train(self):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):
            self.config['optimizer'].zero_grad()

            inputs = batch
            loss, acc, f1 = self.run(inputs)

            if self.args.fp16 == 'True':
                with amp.scale_loss(loss, self.config['optimizer']) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.config['optimizer'].step()
            self.config['scheduler'].step()
            self.progress(loss.data, acc.data, f1)

        return self.return_value()

    def valid(self):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):
                inputs = batch
                loss, acc, f1 = self.run(inputs)

                self.progress(loss.data, acc.data, f1)

        return self.return_value()

    def test(self):
        self.config['model'].load_state_dict(torch.load(self.sorted_path))
        self.config['model'].eval()

        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['test']):
                inputs = batch
                loss, acc, f1 = self.run(inputs, mode='test')

                self.progress(loss.data, acc.data, f1)

        self.metric.cal_rouge(self.hypo, self.memory_db)

        return self.return_value(mode='test')