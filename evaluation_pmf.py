import config
import utils
import pickle
import numpy as np


def ERank(ypred, yreal):
    data_size, model_size = yreal.shape
    ypred_max = np.array([np.nanmax(ypred[i]) for i in range(data_size)])  # 每个data选出最大model

    rank_yreal = np.zeros([data_size, model_size])
    # 对原始数据进行排名，计算排名
    for ii in range(data_size):  # 对每个数据
        rank_yreal[ii] = list(np.sort(yreal[ii])[::-1])  # 排序后再写回（降序）

    # 每个数据推荐的模型的平均排名
    rank = np.zeros(data_size)
    for ii, pred in enumerate(ypred_max):  # 对每个数据
        log1 = rank_yreal[ii] <= pred + 0.0001
        log2 = rank_yreal[ii] >= pred - 0.0001
        if sum(log1 & log2) == 0:
            rank[ii] = model_size
        else:
            rank[ii] = np.where(log1 & log2)[0][0]
    print("AvgRank_pmf: " + str(sum(rank) / data_size / model_size))
    print("AvgRank_pmf: " + str(sum(rank) / data_size))
    return sum(rank) / data_size / model_size, rank


def ETopnk(ypred, yreal, n, k):
    ypred = ypred.T

    data_size, model_size = yreal.shape

    ypred_idx = np.array([np.argsort(ypred[i])[-k:][::-1] for i in range(data_size)])    # 排名前n的
    ypred_max = np.array([np.max(yreal[i][idx]) for i, idx in enumerate(ypred_idx)])
    # ypred_max = np.array([np.nanmax(ypred[i]) for i in range(data_size)])  # 每个data选出最大model

    # 选出实际上的topn
    topn = np.ones((data_size, n))          # data*n
    for ii in range(data_size):             # 对每个数据
        best_value = list(np.sort(yreal[ii])[::-1])         # 第ii列，即第ii个数据（nan已经被填充为0，因此不会被排为最大，当所有数都是nan时，也不会报错）
        topn[ii] = best_value[:n]                           # 第ii个数据的前n个最大值

    correct = 0
    for ii, pred in enumerate(ypred_max):
        for kk, best in enumerate(topn[ii]):
            if pred - 0.0001 <= best <= pred + 0.0001:
                correct += 1
                break

    print("Topnk_pmf:" + str(correct) + '/' + str(data_size))
    return correct/data_size, correct


def main():
    modelnum = 200
    ytrain, ytest, xtrain, xtest = utils.get_data2()
    ytest = ytest[:, :modelnum]
    results = pickle.load(open('results/%s/new/result_pmf_200.pkl' % config.dataset, 'rb'), encoding='iso-8859-1')
    ERank(results['pmf'].T, ytest)
    ETopnk(results['pmf'].T, ytest, int(modelnum*0.05), 1)


if __name__ == '__main__':
    main()