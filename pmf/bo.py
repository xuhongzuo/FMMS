import random

import torch
from pmf import gplvm
import numpy as np
import scipy.stats as st

def ei(fm, fv, ybest, xi=0.01, eps=1e-12):

    fsd = torch.sqrt(fv) + eps
    gamma = (fm - ybest - xi)/fsd

    return fsd * (torch.distributions.Normal(0,1).cdf(gamma) * gamma
                  + torch.distributions.Normal(0,1).log_prob(gamma).exp())

def init_l1(Ytrain, Ftrain, ftest, n_init=5):
    dis = np.abs(Ftrain - ftest).sum(axis=1)
    ix_closest = np.argsort(dis)[:n_init]
    ix_nonnan_pipelines = np.where(np.invert(np.isnan(Ytrain[:,ix_closest].sum(axis=1))))[0]
    ranks = np.apply_along_axis(st.rankdata, 0,
                                Ytrain[ix_nonnan_pipelines[:,None],ix_closest])
    ave_pipeline_ranks = ranks.mean(axis=1)
    ix_init = ix_nonnan_pipelines[np.argsort(ave_pipeline_ranks)[::-1]]

    return ix_init[:n_init]

class BO(gplvm.GP):

    def __init__(self, dim, kernel, acq_func, **kwargs):
        super(BO, self).__init__(dim, np.asarray([]), np.asarray([]), kernel,
                                 **kwargs)

        self.acq_func = acq_func        # ei
        self.ybest = None
        self.xbest = None

    def add(self, xnew, ynew):

        xnew_ = torch.tensor(xnew, dtype=self.X.dtype).reshape((1,-1))
        self.X = torch.cat((self.X, xnew_))
        ynew_ = torch.tensor([ynew], dtype=self.y.dtype)
        self.y = torch.cat((self.y, ynew_))
        if self.ybest is None or ynew_ > self.ybest:
            self.ybest = ynew_
            self.xbest = xnew_
        self.N += 1

    def next(self, Xcandidates):
        # 对新数据，性能非non的模型构成的Ytrain矩阵

        if not self.N:
            return torch.randperm(Xcandidates.size()[0])[0]

        fmean, fvar = self.posterior(Xcandidates)   # 概率矩阵分解？
        # 返回ac函数最大的index
        return torch.argmax(self.acq_func(fmean, fvar, self.ybest))

    def next2(self, Xcandidates):
        # 对新数据，性能非non的模型构成的Ytrain矩阵

        if not self.N:
            # return torch.randperm(Xcandidates.size()[0])[0]
            return torch.tensor([[random.random()] for i in range(len(Xcandidates))])

        fmean, fvar = self.posterior(Xcandidates)   # 概率矩阵分解？
        # 返回ac函数最大的index
        return self.acq_func(fmean, fvar, self.ybest)
