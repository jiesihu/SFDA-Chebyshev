import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def entropy_loss(p, C=2):
    y1 = -1.0*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -torch.mul(input_,torch.log(input_ + epsilon))
    entropy = torch.sum(entropy, dim=1)
    return entropy 
def Entropy_v0(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, logits: torch.tensor, target: torch.tensor, **kwargs):
        return super().forward(logits, target)


class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 1):
        super().__init__()
        self.num_mc_samples = num_mc_samples

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, logits, target, distribution, **kwargs):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        assert num_classes >= 2  # not implemented for binary case with implied background
        # logit_sample = distribution.rsample((self.num_mc_samples,))
        logit_sample = self.fixed_re_parametrization_trick(distribution, self.num_mc_samples)
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)

        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, -1))

        # log_prob = -F.cross_entropy(logit_sample, target, reduction='none').view((self.num_mc_samples, batch_size, -1))
        log_prob = -F.binary_cross_entropy(F.sigmoid(logit_sample), target, reduction='none').view((self.num_mc_samples, batch_size, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(self.num_mc_samples))
        loss = -loglikelihood
        return loss