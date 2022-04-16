import utils
import numpy as np
import pickle


def Test(ytest, speed=1):
    data_size, model_size = ytest.shape
    ypred = np.zeros([data_size, model_size])
    for ii in range(data_size):     # 对每个数据测试
        for jj in range(speed):     # 每个数据rand几次
            pred_idx = np.random.permutation(model_size)[0]
            ypred[ii][pred_idx] = 5 + ytest[ii][pred_idx]
    print(sum(sum(ypred)))
    return ypred


def main():
    import config
    ytrain, ytest, xtrain, xtest = utils.get_data()
    ytest = ytest[:, :2000]
    ypred1 = Test(ytest, speed=1)
    ypred2 = Test(ytest, speed=2)
    ypred4 = Test(ytest, speed=4)
    ypred16 = Test(ytest, speed=16)

    results = pickle.load(open('./results/%s/result_compare.pkl' % config.dataset, 'rb'), encoding='iso-8859-1')
    # results = {}
    results['random1x'] = ypred1
    results['random2x'] = ypred2
    results['random4x'] = ypred4
    results['random16x'] = ypred16
    pickle.dump(results, open('./results/%s/result_compare_2000.pkl' % config.dataset, 'wb'))


if __name__ == '__main__':
    main()