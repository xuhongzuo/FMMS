import config
from MetaOD import MetaODClass
import utils
import pickle
import joblib
import evaluation


def Train(ytrain, yvalid, xtrain, xvalid, txt=''):
    clf = MetaODClass(ytrain, valid_performance=yvalid, n_factors=15,
                      learning='sgd')
    print('training')
    clf.train(n_iter=50, meta_features=xtrain, valid_meta=xvalid,
              learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1,
              n_steps=8)
    joblib.dump(clf, 'models/%s/Metaod_%s.joblib' % (config.dataset, txt))
    return


def Test(xtest, txt=''):
    clf = joblib.load('models/%s/Metaod_%s.joblib' % (config.dataset, txt))
    ypred = clf.predict(xtest)
    return ypred


def main():
    modelnum = 200
    rate = config.get_rate()
    ytrain, ytest, xtrain, xtest = utils.get_data2(rate)
    ytrain = ytrain[:, :modelnum]
    ytest = ytest[:, :modelnum]
    xtrain, ytrain, xvalid, yvalid = utils.train_test_val_split(xtrain, ytrain, rate)
    Train(ytrain, yvalid, xtrain, xvalid)
    ypred = Test(xtest)
    results = {
        'MetaOD': ypred,
    }
    pickle.dump(results, open('./results/%s/result_metaod.pkl' % config.dataset, 'wb'))

    evaluation.ERank(ypred, ytest, 'MetaOD')
    evaluation.ETopnk(ypred, ytest, int(modelnum*0.05), 1, 'MetaOD')


if __name__ == '__main__':
    main()