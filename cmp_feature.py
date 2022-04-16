import numpy as np

import evaluation
import utils
import pickle


def Test(Xtrain, Xtest, Ytrain, Ytest):
    data_size, model_size = Ytest.shape
    ypred = np.zeros([data_size, model_size])
    for ii, xtest in enumerate(Xtest):
        dis = np.abs(Xtrain - xtest).sum(axis=1)        # 按行求和
        closest_idx = np.argsort(dis)[:5]               # 与xtest最相似的5个数据
        best_idx = [np.argsort(Ytrain[id])[-1] for id in closest_idx]
        ypred[ii][best_idx] = np.array([5,4,3,2,1])
    return ypred


def main():
    import config
    ytrain, ytest, xtrain, xtest = utils.get_data2(0.2)
    if config.dataset == 'pmf':
        ytest = ytest[:, :config.modelnum]
        ytrain = ytrain[:, :config.modelnum]
    ypred = Test(xtrain, xtest, ytrain, ytest)
    evaluation.ETopnk(ypred, ytest, int(config.modelnum*0.05), 5)
    evaluation.ERank(ypred, ytest)
    # results = pickle.load(open('./results/%s/result_compare.pkl' % config.dataset, 'rb'), encoding='iso-8859-1')
    # results['feature'] = ypred
    # pickle.dump(results, open('./results/%s/result_compare.pkl'% config.dataset, 'wb'))
    return


if __name__ == '__main__':
    main()