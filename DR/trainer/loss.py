import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss():

    def __init__(self, args):
        self.args = args
        self.softmax = nn.Softmax(dim=-1)

    def base(self, config, logits, labels):

        return config['criterion'](logits, labels.squeeze(1))

    def kd_loss(self, config, post, prior):
        loss_KD = F.kl_div(F.log_softmax(prior / config['args'].temperature, dim=1),
                           F.softmax(post / config['args'].temperature, dim=1), reduction="batchmean")

        return loss_KD

