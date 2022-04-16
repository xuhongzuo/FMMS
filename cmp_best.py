import numpy as np
import evaluation
import utils


def Test(Xtrain, Xtest, Ytrain, Ytest):
    performance = np.sum(Ytrain, axis=0)            # 按列求和（计算每个model的总性能）
    best_idx = np.argsort(performance)[::-1][:5]    # 总performance最大的前5个model的idx

    data_size, model_size = Ytest.shape
    ypred = np.zeros([data_size, model_size])
    for ii, xtest in enumerate(Xtest):
        ypred[ii][best_idx] = np.array([5,4,3,2,1])
    return ypred


def main():
    import config
    ytrain, ytest, xtrain, xtest = utils.get_data2()
    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]

    ypred = Test(xtrain, xtest, ytrain, ytest)
    evaluation.ETopnk(ypred, ytest, int(config.modelnum*0.05), 5)
    evaluation.ERank(ypred, ytest)
    return


if __name__ == '__main__':
    main()