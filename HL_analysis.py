import itertools

import numpy as np
import os as os
import tensorly as tl
import pandas as pd
import plotly.express as px
import pickle
import re
import torch
from HL_tot_cnn import TOTCNN
from HL_regressor import RPCA, RTOT, TOT, ttt
from plotly.subplots import make_subplots
from collections import defaultdict


def show_all_results(sample, egg_channel=0):
    pattern = r'dict_rpes-p=(\d+|\d+\.\d+).p.split'
    m = re.compile(pattern)
    dict_rpes = dict_Bs = list_params = None
    df_tmp = []
    percentile = 0
    Ru = 9
    itr = 16
    # sample = 61
    title_map = dict(
        tot='TOT',
        rpca='RPCA',
        rtot='RTOT',
        cnn='CNN'
    )
    # experiment_root = './experiment-results/lambda-data-{}/'.format(percentile, Ru)
    # model_root = './cnn/model-lambda-data-{}'.format(percentile, Ru)
    experiment_root = './experiment-results/lambda-data-{}/'.format(percentile, Ru)
    model_root = './cnn/model-lambda-data-{}'.format(percentile, Ru)
    # experiment_root = './experiment-results/sync-data-{}/'.format(percentile, Ru)
    # model_root = './cnn/model-sync-data-{}'.format(percentile, Ru)
    for root, dirs, files in os.walk(experiment_root):
        for f in files:
            if 'rpe' in f and m.match(f):
                print(f)
                dict_rpes = pickle.load(open(os.path.join(root, f), 'rb'))
                for model_name, rpes in dict_rpes.items():
                    msg = '{}, mean_rpe={:.6f}, std_rpe={:.6f}, number_of_samples={}'.format(model_name, np.mean(rpes), np.std(rpes), len(rpes))
                    print(msg)
                print('================')
            if 'Bs' in f:
                dict_Bs = pickle.load(open(os.path.join(root, f), 'rb'))
            if 'params' in f:
                list_params = pickle.load(open(os.path.join(root, f), 'rb'))

    params = list_params[itr]
    dims = params['dims']
    pattern = r'.*model-(\d+)-loss_valid=(\d+\.\d+).ckpt'
    m = re.compile(pattern)
    ckpts = [os.path.join(model_root, f) for f in os.listdir(model_root)]
    ckpts = sorted(ckpts, key=lambda x: float(m.match(x)[1]))
    cnn = TOTCNN.load_from_checkpoint(ckpts[itr], **params)

    for model in dict_Bs.keys():
        B = dict_Bs[model][itr]
        y_pre = ttt(params['x_test'], B, params['L'], params['dims'])
        y_test = params['y_test']

        if 'egg' in experiment_root:
            y_test = y_test[:, egg_channel, :]
            y_pre = y_pre[:, egg_channel, :]
            dims = (10, 8, 1, 121)

        y_test = tl.partial_tensor_to_vec(y_test, skip_begin=1)
        y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)
        m_test = np.mean(y_test, axis=1).reshape(-1, 1)
        m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
        y_pre = y_pre - m_pre + m_test

        data = dict(
            y=sum([y_pre[n].flatten().tolist() for n in range(len(y_pre))], []) + sum([y_test[n].flatten().tolist() for n in range(len(y_test))], []),
            x=[i for i in range(1, dims[-1] * dims[-2] + 1)] * 2 * len(y_pre),
            type=[title_map[model] for _ in range(dims[-1] * dims[-2])] * len(y_pre) + ['True' for _ in range(dims[-1] * dims[-2])] * len(y_test),
            sample=[i for i in range(len(y_pre)) for _ in range(dims[-1] * dims[-2])] + [i for i in range(len(y_pre)) for _ in range(dims[-1] * dims[-2])]
        )
        df = pd.DataFrame(data=data)
        df_tmp.append(df[(df['sample'] == sample) & (df['type'] != 'True')])
        df = df[(df['sample'] == sample)]
        fig = px.line(df, x='x', y='y', color='type', line_group='type', line_dash='type')
        fig.update_traces(
            line=dict(width=4)
        )
        fig.update_layout(
            # yaxis_title='Lambda',
            # xaxis_title='Time (0.01 sec)',
            font=dict(
                family="Arial",
                size=42,
                color='black'
            ),
            legend=dict(
                title_font_family="Arial",
                title='Method',
                font=dict(
                    family="Arial",
                    size=42,
                    color="black",

                ),
                bordercolor="black",
                borderwidth=4,
                # bgcolor='antiquewhite',
                # yanchor="top",
                # y=.90,
                # xanchor="left",
                # x=0.39,
                # borderwidth=1
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        fig.update_xaxes(ticks='inside', tickwidth=2, showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(ticks='inside', tickwidth=2, showline=True, linewidth=1, linecolor='black')
        for d in fig.data:
            if d.name == 'True':
                d.line['width'] = 8
            else:
                d.line['width'] = 15
        # fig.show()

    torch.set_grad_enabled(False)
    dims = params['dims']
    cnn.eval()
    y_pre = cnn(torch.tensor(params['x_test']).reshape(-1, 1, dims[0], dims[1]).float()).cpu().numpy()
    y_test = params['y_test']

    if 'egg' in experiment_root:
        y_test = y_test[:, egg_channel, :]
        y_pre = y_pre.squeeze()[:, egg_channel, :]
        dims = (10, 8, 1, 121)

    y_test = tl.partial_tensor_to_vec(y_test, skip_begin=1)
    y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)
    m_test = np.mean(y_test, axis=1).reshape(-1, 1)
    m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
    y_pre = y_pre - m_pre + m_test

    data = dict(
        y=sum([y_pre[n].flatten().tolist() for n in range(len(y_pre))], []) + sum([y_test[n].flatten().tolist() for n in range(len(y_test))], []),
        x=[i for i in range(1, dims[-1] * dims[-2] + 1)] * 2 * len(y_pre),
        type=['CNN' for _ in range(dims[-1] * dims[-2])] * len(y_pre) + ['True' for _ in range(dims[-1] * dims[-2])] * len(y_test),
        sample=[i for i in range(len(y_pre)) for _ in range(dims[-1] * dims[-2])] + [i for i in range(len(y_pre)) for _ in range(dims[-1] * dims[-2])]
    )
    df = pd.DataFrame(data=data)
    df_tmp.append(df[(df['sample'] == sample)])
    # fig = px.line(df, x='x', y='y', color='sample', line_group='type', line_dash='type')
    # fig.show()

    df = pd.concat(df_tmp)
    # df = df[df['type'] == 'True']
    fig = px.line(df, x='x', y='y', color='type', line_dash='type', line_group='type')
    fig.update_traces(
        line=dict(width=5)
    )
    fig.update_layout(
        yaxis_title='',
        xaxis_title='Time (ms)',
        font=dict(
            family="Arial",
            size=42,
            color='black'
        ),
        legend=dict(
            title_font_family="Arial",
            title='',
            font=dict(
                family="Arial",
                size=42,
                color="black",
            ),
            bordercolor="black",
            borderwidth=5,
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    fig.update_xaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black')
    fig.update_yaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black')
    fig.update_layout(showlegend=True)
    for d in fig.data:
        if d.name == 'True':
            d.line['width'] = 5
        else:
            d.line['width'] = 5
    fig.show()

    # y = tl.partial_tensor_to_vec(params['y'], skip_begin=1)
    # data = dict(
    #     y=sum([y[n].flatten().tolist() for n in range(len(y))], []),
    #     time=[i for i in range(1, 204)] * len(y),
    #     sample=[i for i in range(len(y)) for _ in range(203)]
    # )
    # df = pd.DataFrame(data=data)
    # fig = px.line(df, x='time', y='y', color='sample')
    # fig.show()


def plot_lambda_data():
    list_params = pickle.load(open('./experiment-results/lambda-data-0/list_params-p=0.p.split', 'rb'))
    params = list_params[0]
    y_test = params['y_test']
    y_test = tl.partial_tensor_to_vec(y_test, skip_begin=1)
    y_normal = y_test[[12, 14, 16, 17, 18, 23, 33, 34, 35]]
    y_outlier = y_test[[20, 24]]  # y_test[[20, 24, 49]]
    y_test = np.concatenate([y_normal, y_outlier], axis=0)
    n, time = y_test.shape

    data = dict(
        x=[i for i in range(time)] * n,
        y=sum([y_test[i].flatten().tolist() for i in range(n)], []),
        sample=[i for i in range(1, n + 1) for _ in range(time)],
        type=['normal' for _ in range(len(y_normal)) for _ in range(time)] + ['outlier' for _ in range(len(y_outlier)) for _ in range(time)]
    )
    df = pd.DataFrame(data=data)
    fig = px.line(df, x='x', y='y', line_dash='type', line_group='sample', color='type')
    fig.update_traces(
        line=dict(width=5)
    )
    fig.update_layout(
        font=dict(
            family="Arial",
            size=42,
            color='black'
        ),
        legend=dict(
            title_font_family="Arial",
            title='',
            font=dict(
                family="Arial",
                size=42,
                color="black",
            ),
            bordercolor="black",
            borderwidth=5,
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    fig.update_xaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black', title='Time (ms)')
    fig.update_yaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black', title='')
    fig.update_layout(showlegend=True)
    fig.show()
    return


def save_rpe():
    pattern = r'.*-(\d+|\d+\.\d+)-(\d+)|.*-(\d+\.\d+|\d+)'
    m = re.compile(pattern)
    df = {col: [] for col in ['model', 'rpe-mean', 'percentile', 'Ru', 'data-type']}
    for root, dirs, files in os.walk('./experiment-results-2'):
        for f in files:
            if 'rpe' in f and '.p' in f:
                f = os.path.join(root, f)
                match = m.match(f)
                dict_rpes = pickle.load(open(f, 'rb'))
                Ru = 0
                percentile = 0
                for k, v in dict_rpes.items():
                    if 'normal' in root:
                        percentile, Ru, _ = match.groups()
                        df['data-type'].append('sync-data-normal')
                    elif 'sync' in f:
                        percentile, Ru, _ = match.groups()
                        df['data-type'].append('sync-data')
                    elif 'lambda' in f:
                        _, _, percentile = match.groups()
                        df['data-type'].append('lambda-data')
                    elif 'egg' in root:
                        _, _, percentile = match.groups()
                        df['data-type'].append('egg-data')

                    df['percentile'].append(percentile)
                    df['Ru'].append(Ru)
                    df['model'].append(k)
                    df['rpe-mean'].append(str(np.round(np.mean(v), 4)) + '(' + str(np.round(np.std(v), 4)) + ')')

    df = pd.DataFrame(df)
    root = r'./experiment-results-2/'
    for data_type in ['lambda-data', 'sync-data', 'sync-data-normal', 'egg-data']:
        tmp = df[df['data-type'] == data_type]
        tmp = pd.pivot_table(tmp, index=['data-type', 'Ru', 'model'], columns=['percentile'], aggfunc=lambda x: x)
        tmp.to_csv(os.path.join(root, 'rpe-table-{}.csv'.format(data_type)), index=True)
    print(df)


def plot_lambda_input():
    experiment_root = './experiment-results/lambda-data-{}/'.format(0)
    for root, dirs, files in os.walk(experiment_root):
        for f in files:
            if 'params' in f:
                list_params = pickle.load(open(os.path.join(root, f), 'rb'))
    params = list_params[0]
    x = params['x']
    x = x.reshape(-1, 5, 203)
    x = x[:30]
    samples = list(set([i for i in range(30)]) - set([5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 26]))
    x = x[samples]
    n, ch, time = x.shape
    df = dict(
        x=[i for i in range(time)] * (n * ch),
        y=sum([x[i, c, :].flatten().tolist() for i in range(n) for c in range(ch)], []),
        channel=sum([[c for _ in range(time)] for _ in range(n) for c in range(1, ch + 1)], []),
        sample=[i for i in range(n) for _ in range(ch) for _ in range(time)]
    )
    df = pd.DataFrame(df)
    fig = px.line(df, x='x', y='y', line_dash='sample', facet_col='channel')
    fig.update_traces(
        line=dict(width=5)
    )
    fig.update_yaxes(matches=None, showticklabels=True, tickfont={'size': 42})
    fig.update_xaxes(matches='x', tickfont={'size': 42}, title='Time (ms)')
    fig.update_xaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black')
    fig.update_yaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black')
    fig.update_layout(
        font=dict(
            family="Arial",
            size=60,
            color='black'
        ),
        yaxis_title='',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False
    )

    fig.show()
    return


def sensitive_analysis():
    pattern = r'.*-(\d+)-(\d+\.*\d*).*.p.split'
    m = re.compile(pattern)
    root_folder = './sensitive-analysis-aic'
    cols = [str(1), str(10), str(100), str(1000)]  # [str(i) for i in range(1, 30, 1)]
    idx = [str(i) for i in range(2, 11, 1)]
    df = pd.DataFrame(columns=cols, index=idx)

    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if 'aics' in f:
                f = os.path.join(root, f)
                dict_rpes = pickle.load(open(f, 'rb'))
                rpes = [*dict_rpes.values()][0]
                match = m.match(f)
                Ru, mu1 = match[1], match[2]
                if Ru in idx and mu1 in cols:
                    df.loc[Ru, mu1] = np.mean(rpes)
    print(df)
    fig = px.imshow(
        df, x=df.columns, y=df.index,
        aspect='auto', text_auto=False, color_continuous_scale=px.colors.sequential.Cividis_r,
    )
    fig.update_layout(
        font=dict(
            family="Arial",
            size=48,
            color='black'
        ),
        xaxis_title="$\Huge\mu_1 (\\times 10^{-3})$",
        yaxis_title='$\Huge R$',
        coloraxis_colorbar=dict(
            thickness=100,
            # dtick='L0.01',
            nticks=20
        )
    )
    fig.show()
    return


def data_explanation(sample):
    f_param_list = r'C:\Users\jacob\Downloads\TensorToTensorRegression\experiment-results\lambda-data-0\list_params-p=0.p.split'
    f_B = r'C:\Users\jacob\Downloads\TensorToTensorRegression\experiment-results\lambda-data-0\dict_Bs-p=0.p.split'
    param_list = pickle.load(open(f_param_list, 'rb'))
    dict_Bs = pickle.load(open(f_B, 'rb'))
    params = param_list[0]
    B_rpca = dict_Bs['rpca'][0]
    B_roto = dict_Bs['rtot'][0]
    x = params['x']
    y = params['y']
    L = params['L']
    dims = params['dims']
    y_pre_rpca = ttt(x, B_rpca, L, dims)
    y_pre_rtot = ttt(x, B_roto, L, dims)
    s_rpca = y - y_pre_rpca
    s_rtot = y - y_pre_rtot
    y = tl.partial_tensor_to_vec(y, skip_begin=1)
    y_pre_rpca = tl.partial_tensor_to_vec(y_pre_rpca, skip_begin=1)
    y_pre_rtot = tl.partial_tensor_to_vec(y_pre_rtot, skip_begin=1)
    s_rpca = tl.partial_tensor_to_vec(s_rpca, skip_begin=1)
    s_rtot = tl.partial_tensor_to_vec(s_rtot, skip_begin=1)

    df = dict(
        y=y[sample].tolist() + y_pre_rpca[sample].tolist() + y_pre_rtot[sample].tolist() + s_rpca[sample].tolist() + s_rtot[sample].tolist(),
        x=[i for i in range(y.shape[1])] * 5,
        type=['y' for _ in range(y.shape[1])] + ['y_pre' for _ in range(y.shape[1])] * 2 + ['s' for _ in range(y.shape[1])] * 2,
        method=['true' for _ in range(y.shape[1])] + ['RPCA' for _ in range(y.shape[1])] + ['RTOT' for _ in range(y.shape[1])] + ['RPCA' for _ in range(y.shape[1])] + ['RTOT' for _ in
                                                                                                                                                                        range(y.shape[1])]
    )
    df = pd.DataFrame(df)
    fig = px.line(df, x='x', y='y', line_group='method', line_dash='method', color='method', facet_row='type')
    fig.update_traces(
        line=dict(width=5)
    )
    fig.update_yaxes(matches=None, showticklabels=False, tickfont={'size': 42})
    fig.update_xaxes(matches='x', tickfont={'size': 42})
    fig.update_xaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black')
    fig.update_yaxes(ticks='inside', tickwidth=5, showline=True, linewidth=5, linecolor='black')
    fig.update_layout(
        font=dict(
            family="Arial",
            size=48,
            color='black'
        ),
        xaxis_title='Time (ms)',
        legend=dict(
            title_font_family="Arial",
            title='',
            font=dict(
                family="Arial",
                size=42,
                color="black",
            ),
            bordercolor="black",
            borderwidth=5,
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="right",
            x=0.95
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    mapping = dict(y='$\Huge\mathcal{Y}$', y_pre='$\Huge\hat{\mathcal{Y}}$', s='$\Huge\hat{\mathcal{S}}$')
    fig.for_each_yaxis(lambda a: a.update(title=''))
    for name, a in zip(['$\Huge\hat{\mathcal{S}}$', '$\Huge\hat{\mathcal{Y}}$', '$\Huge\mathcal{Y}$'], fig.select_annotations()):
        a.update(text=name, textangle=-0)
    fig.show()
    return


def s_analysis(percentage, ru):
    os.chdir(r'C:\Users\jacob\Downloads\TensorToTensorRegression\experiment-results-1\sync-data-{}-{}'.format(percentage, ru))
    list_params = pickle.load(open('./list_params-p={}.p.split'.format(percentage), 'rb'))
    dict_ss = pickle.load(open('./dict_ss-p={}.p.split'.format(percentage), 'rb'))
    list_s = dict_ss['rtot']
    acc = []
    fnr = []
    fpr = []
    for params, s_est in zip(list_params, list_s):
        y = tl.partial_tensor_to_vec(params['y'], skip_begin=1)
        s = tl.zeros_like(y)
        s_est = tl.partial_tensor_to_vec(s_est, skip_begin=1)
        # print(np.count_nonzero(s_est), s_est.shape)
        # print(len(params['s_idx']))

        tp, tn, fp, fn= 0, 0, 0, 0
        tt = s.size
        if 's_idx' in params:
            for k, v in params['s_idx'].items():
                for i in v:
                    if i < s.shape[1]:
                        s[k][i] = y[k][i]
                tp += np.count_nonzero(s_est[k][[i for i in v if i < s.shape[1]]])
                fn += len(v) - np.count_nonzero(s_est[k][[i for i in v if i < s.shape[1]]])
                fp += np.count_nonzero(s_est[k][~np.array([i for i in v if i < s.shape[1]])])
                tn += len(s_est[k]) - len(v) - np.count_nonzero(s_est[k][~np.array([i for i in v if i < s.shape[1]])])
                # acc.append((tp + tn) / (tp + tn + fp + fn))
                # fnr.append(fn / (tp + fn))
            fp += np.count_nonzero(s_est[[i for i in range(0, s.shape[0]) if i not in params['s_idx']]])
            tn += s_est[[i for i in range(0, s.shape[0]) if i not in params['s_idx']]].size - np.count_nonzero(s_est[[i for i in range(0, s.shape[0]) if i not in params['s_idx']]])
            # print(tt, tp + tn + fp + fn)

        else:
            tp = 0
            tn = s.size - np.count_nonzero(s_est)
            fp = np.count_nonzero(s_est)
            fn = 0

        acc.append((tp + tn) / tt)
        fnr.append(fn / (tp + fn) if percentage > 0 else 0)
        fpr.append(fp / (fp + tn))
    # print(percentage, ru, np.round(np.mean(acc), 2), np.round(np.mean(fnr), 2), np.round(np.mean(fpr), 2), np.linalg.norm(s - s_est) / np.linalg.norm(s))
    return np.round(np.mean(acc), 4), np.round(np.mean(fnr), 4), np.round(np.mean(fpr), 4), np.round(np.linalg.norm(s - s_est) / np.linalg.norm(s), 4)


if __name__ == '__main__':
    # for i in range(0, 20):
    #     show_all_results(i)
    # for i in range(1, 2):
    #     for egg_channel in range(37):
    #         show_all_results(i, egg_channel=egg_channel)
    # plot_lambda_data()
    # plot_lambda_input()
    # sensitive_analysis()
    save_rpe()

    # df = {col: [] for col in ['D', 'R', 'ACC', 'FNR', 'FPR', 'RPE']}
    # for p, ru in itertools.product([0, .03, .05, .1, .15], [5, 7, 9]):
    #     acc, fnr, fpr, rpe = s_analysis(p, ru)
    #     df['D'].append(p)
    #     df['R'].append(ru)
    #     df['ACC'].append(acc)
    #     df['FNR'].append(fnr)
    #     df['FPR'].append(fpr)
    #     df['RPE'].append(rpe)
    # df = pd.DataFrame(df)
    # os.chdir(r'C:\Users\jacob\Downloads\TensorToTensorRegression\experiment-results-1')
    # df.to_csv('./s-analysis-sync.csv', index=False)

    # for i in [104, 115, 117]:
    #     data_explanation(i)
    # print('done')
