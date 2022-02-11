import pandas as pd


def get_mushroom():
    df = pd.read_csv('data/mushroom.csv')
    """ follow https://arxiv.org/pdf/1802.09127.pdf and choose 5e4 datapoints"""
    '''
    df = df.sample(n=50000, replace=True)
    df['index'] = range(50000)
    df = df.set_index('index')
    '''
    df = df.sample(frac=1.0)

    X = df.copy().drop('edible', axis=1)
    y = df.copy()['edible'].astype('category').cat.codes

    X = pd.get_dummies(X)
    print(X.shape)
    return X.to_numpy(), y.to_numpy()
