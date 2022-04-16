dataset = 'pmf'


def get_rate():
    if dataset == 'pmf':
        rate = 0.1
    else:
        rate = 0.2
    return rate


def get_path():
    if dataset == 'pmf':
        fn_data = './data/pmf/target.csv'
        fn_train_ix = './data/pmf/ids_train.csv'
        fn_test_ix = './data/pmf/ids_test.csv'
        fn_data_feats = './data/pmf/feature.csv'
    else:
        fn_data = './data/poc/target.csv'
        fn_train_ix = './data/poc/ids_train.csv'
        fn_test_ix = './data/poc/ids_test.csv'
        fn_data_feats = './data/poc/feature.csv'
    return fn_data, fn_train_ix, fn_test_ix, fn_data_feats