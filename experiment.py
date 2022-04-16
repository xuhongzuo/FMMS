import numpy as np
import pandas as pd

import run_FMUMS
import config
import utils
import evaluation

import torch
import time
import os


def set_para(opt='adam', l='cos', dict1=None, dict2=None):
    if dict1 is None:
        dict1 = {}
    if dict2 is None:
        dict2 = {}
    optlsit = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adagrad': torch.optim.Adagrad}
    losslist = {'rmse': utils.rmse_loss, 'mse': utils.mse_loss,
                'cos': utils.cos_loss, 'L1': utils.l1_loss,
                'sL1': utils.SmoothL1_loss, 'kd': utils.KLDiv_loss
                }
    params = {
        'embedding_size': 4,
        'feature_size': 0,
        'model_size': 0,
        'FM': True,
        'DNN': False,
        'layer_size': 3,
        'hiddensize': 64
    }
    train_params = {
        'batch': 8,
        'lr': 0.001,
        'epoch': 60 if config.dataset == 'poc' else 100,
        'opt': optlsit[opt],
        'optname': opt,
        'loss': losslist[l],
        'lossname': l,
    }
    for key in dict1.keys():
        train_params[key] = dict1[key]
    for key in dict2.keys():
        params[key] = dict2[key]
    return params, train_params


def PMF_effectiveness(runs=5):
    import pickle
    path = './experiments/effectiveness/PMF_%s_runs5.pkl' % config.dataset
    results = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)
    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]

    names = []
    times = results['time']
    epochs = []
    ranks = []
    topns = []
    topn_nums = []
    detailranks = []

    for r in range(runs):
        names.append('pmf_%d' % r)
        ypred = results['pmf_%d' % r]  # 第r次的结果
        rank, detailrank = evaluation.ERank(ypred, ytest)  # 所推荐的模型的平均排名
        topn, num = evaluation.ETopnk(ypred, ytest, int(config.modelnum * 0.05), 5, 'pmf')  # 所推荐的模型中，排在前n个的里有几个

        ranks.append(rank)
        topns.append(topn)
        topn_nums.append(num)
        epochs.append(0)
        detailranks.append(detailrank)
    results = {
        'names': names,
        'times': times,
        'epochs': epochs,
        'ranks': ranks,
        'topns': topns,
        'topn_num': topn_nums,
    }

    effect_path = './experiments/effectiveness/PMF_%s_runs5.csv' % config.dataset
    if os.path.exists(effect_path):  # 如果路径存在，则在后面罗结果
        df = pd.read_csv(effect_path)
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='outer')  # 摞起来
        results.to_csv(effect_path, index=False)
    else:  # 否则新建文件
        results = pd.DataFrame(results)
        results.to_csv(effect_path, index=False)

    rank_path = "./experiments/detailresult/result_PMF_%s.csv" % config.dataset
    detailranks = pd.DataFrame(detailranks)
    detailranks['model'] = ['pmf' for i in range(5)]
    detailranks.to_csv(rank_path, index=False)
    return


def Feature_effectiveness(runs=5):
    import cmp_feature
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)

    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]

    names = []
    times = []
    epochs = []
    ranks = []
    topns = []
    topn_nums = []
    detailranks = []

    for r in range(runs):
        print('running %d.....' % r)
        rpath = "feature_%s_%s_%s" % (config.dataset, str(config.modelnum), str(r))  # pmf_para_runs
        names.append(rpath)

        start = time.time()
        ypred = cmp_feature.Test(xtrain, xtest, ytrain, ytest)
        end = time.time()
        rank, detailrank = evaluation.ERank(ypred, ytest, 'feature')  # 所推荐的模型的平均排名
        topn, num = evaluation.ETopnk(ypred, ytest, int(config.modelnum * 0.05), 5, 'feature')  # 所推荐的模型中，排在前n个的里有几个

        times.append(end - start)
        ranks.append(rank)
        topns.append(topn)
        topn_nums.append(num)
        epochs.append(0)
        detailranks.append(detailrank)
    results = {
        'names': names,
        'times': times,
        'epochs': epochs,
        'ranks': ranks,
        'topns': topns,
        'topn_num': topn_nums,
    }

    effect_path = './experiments/effectiveness/Feature_%s_runs5.csv' % config.dataset
    if os.path.exists(effect_path):  # 如果路径存在，则在后面罗结果
        df = pd.read_csv(effect_path)
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='outer')  # 摞起来
        results.to_csv(effect_path, index=False)
    else:  # 否则新建文件
        results = pd.DataFrame(results)
        results.to_csv(effect_path, index=False)

    rank_path = "./experiments/detailresult/result_Feature_%s.csv" % config.dataset
    if os.path.exists(rank_path):
        df = pd.read_csv(rank_path)
        columns = [str(i) for i in range(xtest.shape[0])]
        detailranks = pd.DataFrame(detailranks, columns=columns)
        detailranks['model'] = ['Feature' for i in range(5)]
        detailranks = pd.concat([df, detailranks], axis=0, ignore_index=True)
        detailranks.to_csv(rank_path, index=False)
    else:  # 否则新建文件
        detailranks = pd.DataFrame(detailranks)
        detailranks['model'] = ['Feature' for i in range(5)]
        detailranks.to_csv(rank_path, index=False)


def Best_effectiveness(runs=5):
    import cmp_best
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)

    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]

    names = []
    times = []
    epochs = []
    ranks = []
    topns = []
    topn_nums = []
    detailranks = []

    for r in range(runs):
        print('running %d.....' % r)
        rpath = "best_%s_%s_%s" % (config.dataset, str(config.modelnum), str(r))  # pmf_para_runs
        names.append(rpath)

        start = time.time()
        ypred = cmp_best.Test(xtrain, xtest, ytrain, ytest)
        end = time.time()
        rank, detailrank = evaluation.ERank(ypred, ytest, 'best')  # 所推荐的模型的平均排名
        topn, num = evaluation.ETopnk(ypred, ytest, int(config.modelnum * 0.05), 5, 'best')  # 所推荐的模型中，排在前n个的里有几个

        times.append(end - start)
        ranks.append(rank)
        topns.append(topn)
        topn_nums.append(num)
        epochs.append(0)
        detailranks.append(detailrank)
    results = {
        'names': names,
        'times': times,
        'epochs': epochs,
        'ranks': ranks,
        'topns': topns,
        'topn_num': topn_nums,
    }

    effect_path = './experiments/effectiveness/Best_%s_runs5.csv' % config.dataset
    if os.path.exists(effect_path):  # 如果路径存在，则在后面罗结果
        df = pd.read_csv(effect_path)
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='outer')  # 摞起来
        results.to_csv(effect_path, index=False)
    else:  # 否则新建文件
        results = pd.DataFrame(results)
        results.to_csv(effect_path, index=False)

    rank_path = "./experiments/detailresult/result_Best_%s.csv" % config.dataset
    if os.path.exists(rank_path):
        df = pd.read_csv(rank_path)
        columns = [str(i) for i in range(xtest.shape[0])]
        detailranks = pd.DataFrame(detailranks, columns=columns)
        detailranks['model'] = ['best' for i in range(5)]
        detailranks = pd.concat([df, detailranks], axis=0, ignore_index=True)
        detailranks.to_csv(rank_path, index=False)
    else:  # 否则新建文件
        detailranks = pd.DataFrame(detailranks)
        detailranks['model'] = ['best' for i in range(5)]
        detailranks.to_csv(rank_path, index=False)


def Rand_effectiveness(speed, runs=5):
    import cmp_rand
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)

    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]

    names = []
    times = []
    epochs = []
    ranks = []
    topns = []
    topn_nums = []
    detailranks = []

    for r in range(runs):
        print('running %d.....' % r)
        rpath = "Rand_%s_%s_%s_%s" % (config.dataset, str(config.modelnum), str(speed), str(r))  # pmf_para_runs
        names.append(rpath)

        start = time.time()
        ypred = cmp_rand.Test(ytest, speed=speed)
        end = time.time()
        rank, detailrank = evaluation.ERank(ypred, ytest, 'Rand')  # 所推荐的模型的平均排名
        topn, num = evaluation.ETopnk(ypred, ytest, int(config.modelnum * 0.05), 5, 'Rand')  # 所推荐的模型中，排在前n个的里有几个

        times.append(end - start)
        ranks.append(rank)
        topns.append(topn)
        topn_nums.append(num)
        epochs.append(0)
        detailranks.append(detailrank)
    results = {
        'names': names,
        'times': times,
        'epochs': epochs,
        'ranks': ranks,
        'topns': topns,
        'topn_num': topn_nums,
    }

    effect_path = './experiments/effectiveness/Rand_%s_runs5.csv' % config.dataset
    if os.path.exists(effect_path):     # 如果路径存在，则在后面罗结果
        df = pd.read_csv(effect_path)
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='outer')    # 摞起来
        results.to_csv(effect_path, index=False)
    else:                               # 否则新建文件
        results = pd.DataFrame(results)
        results.to_csv(effect_path, index=False)

    rank_path = "./experiments/detailresult/result_Rand_%s.csv" % config.dataset
    if os.path.exists(rank_path):
        df = pd.read_csv(rank_path)
        columns = [str(i) for i in range(xtest.shape[0])]
        detailranks = pd.DataFrame(detailranks, columns=columns)
        detailranks['model'] = ['Rand_%s' % str(speed) for i in range(5)]
        detailranks = pd.concat([df, detailranks], axis=0, ignore_index=True)
        detailranks.to_csv(rank_path, index=False)
    else:  # 否则新建文件
        detailranks = pd.DataFrame(detailranks)
        detailranks['model'] = ['Rand_%s' % str(speed) for i in range(5)]
        detailranks.to_csv(rank_path, index=False)


def MetaOD_effectiveness(runs=5):
    import cmp_metaod
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)

    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]
    xtrain, ytrain, xvalid, yvalid = utils.train_test_val_split(xtrain, ytrain, rate)

    names = []
    times = []
    epochs = []
    ranks = []
    topns = []
    topn_nums = []
    detailranks = []

    for r in range(runs):
        print('running %d.....' % r)
        rpath = "MetaOD_%s_%s_%s" % (config.dataset, str(config.modelnum), str(r))  # pmf_para_runs
        names.append(rpath)

        start = time.time()
        cmp_metaod.Train(ytrain, yvalid, xtrain, xvalid, str(r))
        end = time.time()
        ypred = cmp_metaod.Test(xtest, str(r))
        rank, detailrank = evaluation.ERank(ypred, ytest, 'MetaOD')  # 所推荐的模型的平均排名
        topn, num = evaluation.ETopnk(ypred, ytest, int(config.modelnum * 0.05), 5, 'MetaOD')  # 所推荐的模型中，排在前n个的里有几个

        times.append(end-start)
        ranks.append(rank)
        topns.append(topn)
        topn_nums.append(num)
        epochs.append(0)
        detailranks.append(detailrank)
    results = {
        'names': names,
        'times': times,
        'epochs': epochs,
        'ranks': ranks,
        'topns': topns,
        'topn_num': topn_nums,
    }

    effect_path = './experiments/effectiveness/MetaOD_%s_runs5.csv' % config.dataset
    if os.path.exists(effect_path):     # 如果路径存在，则在后面罗结果
        df = pd.read_csv(effect_path)
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='outer')    # 摞起来
        results.to_csv(effect_path, index=False)
    else:                               # 否则新建文件
        results = pd.DataFrame(results)
        results.to_csv(effect_path, index=False)

    rank_path = "./experiments/detailresult/result_MetaOD_%s.csv" % config.dataset
    if os.path.exists(rank_path):
        df = pd.read_csv(rank_path)
        columns = [str(i) for i in range(xtest.shape[0])]
        detailranks = pd.DataFrame(detailranks, columns=columns)
        detailranks['model'] = ['MetaOD' for i in range(5)]
        detailranks = pd.concat([df, detailranks], axis=0, ignore_index=True)
        detailranks.to_csv(rank_path, index=False)
    else:  # 否则新建文件
        detailranks = pd.DataFrame(detailranks)
        detailranks['model'] = ['MetaOD' for i in range(5)]
        detailranks.to_csv(rank_path, index=False)


def FMUMS_effectiveness(runs=5, loss='cos'):
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)

    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]

    params, train_params = set_para('adam', loss)
    params['feature_size'] = xtrain.shape[1]
    params['model_size'] = ytrain.shape[1]

    xtrain, ytrain, xvalid, yvalid = utils.train_test_val_split(xtrain, ytrain, rate)

    names = []
    times = []
    ranks = []
    topns = []
    topn_nums = []
    epochs = []
    detailranks = []

    path = config.get_para(train_params, params)  # 参数名
    for r in range(runs):
        print('running %d.....' % r)
        rpath = "%s_%s_%s_%s" % (config.dataset, path, str(config.modelnum), str(r))  # pmf_para_runs
        names.append(rpath)

        start = time.time()
        log = run_FMUMS.Train(params, train_params, xtrain, xvalid, xtest, ytrain, yvalid, ytest, str(r))
        end = time.time()

        # log.to_csv('./experiments/convergence/FMUMS_%s_%s_runs5_%s.csv' % (config.dataset, path, str(r)))      # FMUMS在pmf数据集上参数为path时第r次实验的中间结果

        ypred = run_FMUMS.Test(params, train_params, xtest, ytest, str(r))
        rank, detailrank = evaluation.ERank(ypred, ytest, 'FMUMS')  # 所推荐的模型的平均排名
        topn, num = evaluation.ETopnk(ypred, ytest, int(config.modelnum * 0.05), 5, 'FMUMS')  # 所推荐的模型中，排在前n个的里有几个

        times.append(end-start)
        ranks.append(rank)
        topns.append(topn)
        topn_nums.append(num)
        epochs.append(len(log['loss_train_set']))
        detailranks.append(detailrank)
    results = {
        'names': names,
        'times': times,
        'epochs': epochs,
        'ranks': ranks,
        'topns': topns,
        'topn_num': topn_nums,
    }

    effect_path = './experiments/effectiveness/FMUMS_%s_runs5.csv' % config.dataset
    if os.path.exists(effect_path):     # 如果路径存在，则在后面罗结果
        df = pd.read_csv(effect_path)
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='outer')    # 摞起来
        results.to_csv(effect_path, index=False)
    else:                               # 否则新建文件
        results = pd.DataFrame(results)
        results.to_csv(effect_path, index=False)

    rank_path = "./experiments/detailresult/result_FMUMS_%s.csv" % config.dataset
    if os.path.exists(rank_path):
        df = pd.read_csv(rank_path)
        columns = [str(i) for i in range(xtest.shape[0])]
        detailranks = pd.DataFrame(detailranks, columns=columns)
        detailranks['model'] = [path for i in range(5)]
        detailranks = pd.concat([df, detailranks], axis=0, ignore_index=True)
        detailranks.to_csv(rank_path, index=False)
    else:  # 否则新建文件
        detailranks = pd.DataFrame(detailranks)
        detailranks['model'] = [path for i in range(5)]
        detailranks.to_csv(rank_path, index=False)


def FMUMS_getrank(loss):
    import pickle
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)
    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]
    params, train_params = set_para('adam', loss)
    params['feature_size'] = xtrain.shape[1]
    params['model_size'] = ytrain.shape[1]
    path = config.get_para(train_params, params)  # 参数名

    detailranks = []
    for r in range(5):
        ypred = pickle.load(open('./results/%s/result_FMUMS_%s_%s.pkl' % (config.dataset, path, str(r)), 'rb'), encoding='iso-8859-1')['FMUMS']
        _, detailrank = evaluation.ERank(ypred, ytest)
        detailranks.append(detailrank)

    rank_path = "./experiments/detailresult/result_FMUMS_%s.csv" % config.dataset
    if os.path.exists(rank_path):
        df = pd.read_csv(rank_path)
        columns = [str(i) for i in range(xtest.shape[0])]
        detailranks = pd.DataFrame(detailranks, columns=columns)
        detailranks['model'] = [path for i in range(5)]
        detailranks = pd.concat([df, detailranks], axis=0, ignore_index=True)
        detailranks.to_csv(rank_path, index=False)
    else:  # 否则新建文件
        detailranks = pd.DataFrame(detailranks)
        detailranks['model'] = [path for i in range(5)]
        detailranks.to_csv(rank_path, index=False)


def FMUMS_parameter(dict1=None, dict2=None, runs=5):
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)

    if config.dataset == 'pmf':
        ytrain = ytrain[:, :config.modelnum]
        ytest = ytest[:, :config.modelnum]

    params, train_params = set_para('adam', 'cos', dict1, dict2)
    params['feature_size'] = xtrain.shape[1]
    params['model_size'] = ytrain.shape[1]

    xtrain, ytrain, xvalid, yvalid = utils.train_test_val_split(xtrain, ytrain, rate)

    names = []
    times = []
    ranks = []
    topns = []
    topn_nums = []
    epochs = []
    detailranks = []

    path = config.get_para(train_params, params)  # 参数名
    for r in range(runs):
        print('running %d.....' % r)
        rpath = "%s_%s_%s_%s" % (config.dataset, path, str(config.modelnum), str(r))  # pmf_para_runs
        names.append(rpath)

        start = time.time()
        log = run_FMUMS.Train(params, train_params, xtrain, xvalid, xtest, ytrain, yvalid, ytest, str(r))
        end = time.time()

        # log.to_csv('./experiments/convergence/FMUMS_%s_%s_runs5_%s.csv' % (
        # config.dataset, path, str(r)))  # FMUMS在pmf数据集上参数为path时第r次实验的中间结果

        ypred = run_FMUMS.Test(params, train_params, xtest, ytest, str(r))
        rank, detailrank = evaluation.ERank(ypred, ytest, 'FMUMS')  # 所推荐的模型的平均排名
        topn, num = evaluation.ETopnk(ypred, ytest, int(config.modelnum * 0.05), 5, 'FMUMS')  # 所推荐的模型中，排在前n个的里有几个

        times.append(end - start)
        ranks.append(rank)
        topns.append(topn)
        topn_nums.append(num)
        epochs.append(len(log['loss_train_set']))
        detailranks.append(detailrank)
    results = {
        'names': names,
        'times': times,
        'epochs': epochs,
        'ranks': ranks,
        'topns': topns,
        'topn_num': topn_nums,
    }

    effect_path = './experiments/parameter/FMUMS_%s_runs5.csv' % config.dataset
    if os.path.exists(effect_path):  # 如果路径存在，则在后面罗结果
        df = pd.read_csv(effect_path)
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='outer')  # 摞起来
        results.to_csv(effect_path, index=False)
    else:  # 否则新建文件
        results = pd.DataFrame(results)
        results.to_csv(effect_path, index=False)

    rank_path = "./experiments/detailresult/result_FMUMS_%s.csv" % config.dataset
    if os.path.exists(rank_path):
        df = pd.read_csv(rank_path)
        columns = [str(i) for i in range(xtest.shape[0])]
        detailranks = pd.DataFrame(detailranks, columns=columns)
        detailranks['model'] = [path for i in range(5)]
        detailranks = pd.concat([df, detailranks], axis=0, ignore_index=True)
        detailranks.to_csv(rank_path, index=False)
    else:  # 否则新建文件
        detailranks = pd.DataFrame(detailranks)
        detailranks['model'] = [path for i in range(5)]
        detailranks.to_csv(rank_path, index=False)


def test_time(model_name):
    times = []
    features = [20, 40, 60, 80, 100]
    models = [100, 200, 400, 800, 1600]
    datas = [100, 200, 400, 800, 1600]

    para = {
        'features': [],
        'models': [],
        'datas': []
    }
    # features = [20]
    # models = [100, 200]
    # datas = [50, 100]

    def run(data_size, model_size, feature_size):
        para['features'].append(feature_size)
        para['models'].append(model_size)
        para['datas'].append(data_size)

        Y = np.random.random([data_size, model_size])
        X = np.random.random([data_size, feature_size])

        xtrain, ytrain, xtest, ytest = utils.train_test_val_split(X, Y, 0.1)
        xtrain, ytrain, xvalid, yvalid = utils.train_test_val_split(xtrain, ytrain, 0.1)

        params, train_params = set_para('adam', 'cos')
        params['feature_size'] = feature_size
        params['model_size'] = model_size

        start = time.time()
        if model_name == 'FMUMS':
            run_FMUMS.Train(params, train_params, xtrain, xvalid, xtest, ytrain, yvalid, ytest)
        elif model_name == 'MetaOD':
            import cmp_metaod
            cmp_metaod.Train(ytrain, yvalid, xtrain, xvalid)
        elif model_name == 'PMF':
            from pmf import pmf_zry
            pmf_zry.run(ytrain.T, ytest.T, xtrain, xtest)
        else:
            print('error')
        end = time.time()

        return end-start

    for data_size in datas:
        model_size = 200
        feature_size = 50
        timecost = run(data_size, model_size, feature_size)
        times.append(timecost)

    np.savetxt('out.txt', np.array([times]), fmt="%.4f", delimiter=',')

    for feature_size in features:
        data_size = 500
        model_size = 200
        timecost = run(data_size, model_size, feature_size)
        times.append(timecost)
    np.savetxt('out.txt', np.array([times]), fmt="%.4f", delimiter=',')

    for model_size in models:
        data_size = 500
        feature_size = 50
        timecost = run(data_size, model_size, feature_size)
        times.append(timecost)
    np.savetxt('out.txt', np.array([times]), fmt="%.4f", delimiter=',')

    path = 'experiments/efficiency%s.csv' % config.dataset
    if os.path.exists(path):
        times_df = pd.read_csv(path)
        times_df[model_name] = times
        times_df.to_csv(path, index=False)
    else:  # 否则新建文件
        times_df = pd.DataFrame(para)
        times_df[model_name] = times
        times_df.to_csv(path, index=False)


if __name__ == '__main__':
    test_time('MetaOD')

    # FMUMS
    # paratest
    ks = [1, 2, 4, 8, 16, 32]
    batchs = [2, 4, 8, 16, 32, 64]
    lrs = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    epochs = [20, 40, 60, 80, 100, 120]
    # for batch in batchs:
    #     FMUMS_parameter(dict1={'batch': batch})
    # for k in ks:
    #     FMUMS_parameter(dict2={'embedding_size': k})
    # for lr in lrs:
    #     FMUMS_parameter(dict1={'lr': lr})
    # for epoch in epochs:
    #     FMUMS_parameter(dict1={'epoch': epoch})

    # FMUMS
    # loss_list = ['cos', 'rmse', 'mse', 'L1', 'sL1', 'kd']
    # loss_list = ['cos']
    # for loss in loss_list:
    #     FMUMS_effectiveness(loss=loss)
    # for loss in loss_list:
    #     FMUMS_getrank(loss=loss)

    # MetaOD
    # MetaOD_effectiveness()

    # PMF
    # PMF_effectiveness()

    # Rand
    # for speed in [1, 2, 4]:
    #     Rand_effectiveness(speed)
    #
    # # Feature
    # Feature_effectiveness()
    #
    # # Best
    # Best_effectiveness()