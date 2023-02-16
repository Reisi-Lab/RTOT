import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as sparse
import tensorly as tl

from scipy.io import loadmat
from collections import defaultdict


def ttt(x, b, L, dims):
    return np.tensordot(
        x,
        b,
        axes=[
            [k + 1 for k in range(len(dims[:L]))],
            [k for k in range(len(dims[:L]))]
        ]
    )


def gen_lambda_data(**params):
    N = params['N'] = 200
    params['L'] = 2
    params['M'] = 2
    params['dims'] = (35, 29, 7, 29)

    data_lambda = loadmat('./data/clustered_data_lambda.mat')
    x = data_lambda['clustered_input'][0]
    y = data_lambda['clustered_output'][0]
    idx = list(set([i for i in range(len(x))]) - set([2, 3, 10, 13, 14, 16]))
    tmpx = []
    tmpy = []

    for rx, ry in zip(x[idx], y[idx]):
        tmpy.append(ry)
        tmpx.append(rx)

    x = np.concatenate(tmpx, axis=0).reshape(-1, 35, 29)
    y = np.concatenate(tmpy, axis=0).reshape(-1, 7, 29)

    x_train, x_test = x[:N], x[N:]
    y_train, y_test = y[:N], y[N:]

    params['x'] = x_train
    params['y'] = y_train
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params


def mcp(x, a=1, lam=.8):
    return lam * abs(x) - x ** 2 / (2 * a) if abs(x) < a * lam else (a * lam ** 2) / 2


def gen_sync_data(**params):
    N = params['N'] = 200
    L = params['L'] = 2
    M = params['M'] = 2
    Ru = params['Ru']
    dims = params['dims'] = (35, 29, 7, 29)
    percentile = params['percentile']
    R = 5  # np.random.randint(low=3, high=7)

    x = [
        [
            np.random.uniform(1e-3, 1) * np.concatenate(
                [np.array([np.cos(1e-3 * np.pi * r * (j / p)) if r % 2 == 1 else np.sin(1e-3 * np.pi * r * (j / p)) for j in range(1, p + 1)]).reshape(-1, 1) for r in range(1, R + 1)],
                axis=1
            )
            for p in dims[:2]
        ] for _ in range(1, N + 1)
    ]
    x = [tl.cp_to_tensor((None, t)) for t in x]

    rns = np.random.RandomState(seed=8)
    b = [rns.normal(size=(p, Ru)) for p in dims]
    b = tl.cp_to_tensor((None, b))
    b = b / np.linalg.norm(b)
    y = ttt(x, b, L, dims)

    b2 = np.zeros_like(b)
    b2 = b2.flatten()
    b2[:50] = np.random.normal(loc=0, scale=1, size=50)
    b2 = (b2 / np.linalg.norm(b2)).reshape(dims)
    y += ttt(x, b2, L, dims)

    e = np.random.normal(loc=0, scale=1, size=y.shape)
    e = e / np.linalg.norm(e)
    y += e

    # y = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': y.flatten(), 'time': [i for i in range(203)] * N, 'n': [i for i in range(N) for _ in range(203)]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()

    x_test, x = x[:80], x[80:]
    y_test, y = y[:80], y[80:]

    if percentile > 0:
        percentage = percentile
        sample_indices = np.random.randint(0, 120, size=int(percentage * 120))
        outlier_idx = {i: [] for i in sample_indices}
        y_with_outliers = tl.partial_tensor_to_vec(y[sample_indices], skip_begin=1)
        for idx, sample in zip(sample_indices, y_with_outliers):
            start_index = np.random.randint(100, 194)
            outlier_length = np.random.randint(10, 30)
            outlier_idx[idx] = [i for i in range(start_index, start_index + outlier_length)]
            sample[start_index: start_index + outlier_length] = np.random.uniform(.8, 2, size=min(outlier_length, 203 - start_index))
        y[sample_indices] = y_with_outliers.reshape(-1, params['dims'][2], params['dims'][3])
        params['s_idx'] = outlier_idx

    # tmp = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': tmp.flatten(), 'time': [i for i in range(203)] * 120, 'n': [i for i in range(120) for _ in range(203)]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()
    # fig.write_html('./training-data.html')

    params['x'] = x
    params['y'] = y
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params


def gen_sync_data_norm(**params):
    N = params['N'] = 500
    L = params['L'] = 2
    M = params['M'] = 2
    dims = params['dims'] = (15, 20, 5, 10)
    percentile = params['percentile']
    Ru = params['Ru']
    scale = params['scale']

    rns = np.random.RandomState(seed=1) # normal seed=1
    base = [rns.normal(size=(p, 5)) for p in dims[:L]]
    x = [[np.random.uniform(1e-3, 1) * b for b in base] for _ in range(1, N + 1)]
    x = [tl.cp_to_tensor((None, t)) for t in x]

    # rns = np.random.RandomState(seed=5) this one is for normal
    rns = np.random.RandomState(seed=8)
    b = [rns.normal(size=(p, Ru)) for p in dims]
    b = tl.cp_to_tensor((None, b))
    b = b / np.linalg.norm(b)
    y = ttt(x, b, L, dims)

    e = np.random.normal(loc=0, scale=1, size=y.shape)
    e = e / np.linalg.norm(e)
    y += e

    # y = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': y.flatten(), 'time': [i for i in range(dims[-1]*dims[-2])] * N, 'n': [i for i in range(N) for _ in range(dims[-1]*dims[-2])]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()

    x_test, x = x[:100], x[100:]
    y_test, y = y[:100], y[100:]

    if percentile > 0:
        y = tl.partial_tensor_to_vec(y, skip_begin=1)
        indices = np.random.randint(0, 400, size=int(400 * percentile))
        outlier_idx = {i: [] for i in indices}
        # y_with_outlier = y[indices]
        for n in indices:
            idx = np.random.randint(0, 40)
            outlier_idx[n] = [i for i in range(idx, idx + 5)]
            # y_[idx] = scale * np.sign(y_[idx]) * max(y_)
            y[n][idx: idx + 5] = np.random.uniform(.8, 2, size=5)
        # y[indices] = y_with_outlier
        y = y.reshape(-1, dims[-2], dims[-1])
        params['s_idx'] = outlier_idx


    # tmp = tl.partial_tensor_to_vec(y, skip_begin=1)
    # df = {'y': tmp.flatten(), 'time': [i for i in range(dims[-1]*dims[-2])] * 400, 'n': [i for i in range(400) for _ in range(dims[-1]*dims[-2])]}
    # fig = px.line(df, x='time', y='y', line_group='n', color='n')
    # fig.show()
    # # fig.write_html('./training-data.html')
    #
    params['x'] = x
    params['y'] = y
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params


def gen_egg_data(**params):
    eeg = loadmat('./EGGData/EEG_Processed_Data.mat')
    fmri = loadmat('./EGGData/fMRI_Data_preproc.mat')
    data_eeg = eeg['mean_value'].T
    data_eeg = np.concatenate((data_eeg[:4, :, :], data_eeg[5:, :, :]), axis=0)
    data_fmri = fmri['ROI_DATA'].reshape(16, 10, 8)
    y, x = data_eeg, data_fmri

    N = params['N'] = 14
    params['L'] = 2
    params['M'] = 2
    params['dims'] = (10, 8, 37, 121)

    x_train, x_test = x[:N], x[N:]
    y_train, y_test = y[:N], y[N:]

    params['x'] = x_train
    params['y'] = y_train
    params['x_test'] = x_test
    params['y_test'] = y_test

    return params


if __name__ == '__main__':
    # params = dict()
    # params = gen_lambda_data(**params)
    # x, y = params['x_test'], params['y_test']
    #
    # arr = np.array([y.flatten(), [i for i in range(1, 204)] * y.shape[0], [n for n in range(1, y.shape[0] + 1) for _ in range(203)]]).T
    # df = pd.DataFrame(data=arr, columns=['y', 'time', 'type'])
    # fig = px.line(df, x='time', y='y', line_group='type', color='type')
    # fig.show()
    # print(x.shape, y.shape)
    # print('done')

    params = dict(
        R=15,
        Ru=3,
        mu1=6.5e-3,
        mu2=3.5e-3,
        mu3=1e-8,
        tol=1e-4,
        max_itr=20,
        replications=20,
        percentile=.15,
        scale=2
    )
    # gen_sync_data(**params)
    params = gen_sync_data_norm(**params)
    print(params['s_idx'])
    # gen_lambda_data(**params)
    # gen_egg_data(**params)
