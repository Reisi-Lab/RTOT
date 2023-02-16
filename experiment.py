import HL_data_io as io
import numpy as np
import os as os
import tensorly as tl
import pandas as pd
import plotly.express as px
import pickle
import time
from HL_regressor import RPCA, RTOT, TOT, ttt
from collections import defaultdict
from itertools import product
from functools import reduce
from HL_rtot_split import RTOTSplit

if __name__ == '__main__':
    replications = 10

    for percentile, Ru, mu1 in product([.1], [7, 10, 20, 30], [9e-3]):
        folder = r'./computational-time-3/sync-data-normal-{}-{}/'.format(percentile, Ru)
        dict_time = defaultdict(list)
        dict_Bs = defaultdict(list)
        dict_rpes = defaultdict(list)
        dict_aics = defaultdict(list)
        dict_ss = defaultdict(list)
        list_params = []
        # f = r'C:\Users\jacob\Downloads\TensorToTensorRegression\experiment-results\sync-data-normal-0.05-5\list_params-p=0.05.p.split'
        # params = pickle.load(open(f, 'rb'))[0]
        # print(params['mu1'])
        for r in range(replications):
            params = dict(
                R=Ru,
                Ru=5,  # Ru,
                mu1=mu1,  # 9e-3,
                mu2=1.5e-3,
                mu3=1.5e-3, # 1e-10,
                tol=1e-6,
                max_itr=20,
                replications=replications,
                percentile=percentile,
                scale=10
            )
            # params = io.gen_sync_data(**params)
            # params = io.gen_lambda_data(**params)
            params = io.gen_sync_data_norm(**params)
            # params = io.gen_egg_data(**params)
            # params['R'] = Ru
            # params['mu1'] = mu1 * 1e-3
            list_params.append(params)

            # print('============')
            for model in [TOT(**params), RTOT(**params), RPCA(**params)]:
                start = time.time()
                ja, B, AIC, s = model.fit(verbose=False)
                end = time.time()

                y_test = tl.partial_tensor_to_vec(params['y_test'], skip_begin=1)
                y_pre = ttt(params['x_test'], B, params['L'], params['dims'])
                y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)
                m_test = np.mean(y_test, axis=1).reshape(-1, 1)
                m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
                y_pre = y_pre - m_pre + m_test

                # data = dict(
                #     y=sum([y_pre[n].flatten().tolist() for n in range(len(y_pre))], []) + sum([y_test[n].flatten().tolist() for n in range(len(y_test))], []),
                #     time=[i for i in range(1, 204)] * 2 * len(y_pre),
                #     type=[model.name for _ in range(203)] * len(y_pre) + ['true' for _ in range(203)] * len(y_test),
                #     sample=[i for i in range(len(y_pre)) for _ in range(203)] + [i for i in range(len(y_pre)) for _ in range(203)]
                # )
                # df = pd.DataFrame(data=data)
                # fig = px.line(df, x='time', y='y', color='sample', line_group='type', line_dash='type')
                # fig.show()

                rpe = np.linalg.norm(y_pre - y_test) / np.linalg.norm(y_test)
                dict_Bs[model.name].append(B)
                dict_rpes[model.name].append(rpe)
                dict_aics[model.name].append(AIC)
                dict_ss[model.name].append(s)
                dict_time[model.name].append(end - start)

                # msg = 'ru={}, replication={}, model={}, rpe={:.4f}, aic={:.4f}, time={:.4f}'.format(Ru, r + 1, model.name, rpe, AIC, end - start)
                # print(msg)
        print('============')
        for k, v in dict_time.items():
            print('ru={}, model={}, avg time={:.4f}'.format(Ru, k, np.mean(v)))
        print('============')

        if not os.path.exists(folder): os.makedirs(folder)
        pickle.dump(list_params, open(os.path.join(folder, 'list_params-p={}.p.split'.format(percentile)), 'wb'))
        pickle.dump(dict_Bs, open(os.path.join(folder, 'dict_Bs-p={}.p.split'.format(percentile)), 'wb'))
        pickle.dump(dict_rpes, open(os.path.join(folder, 'dict_rpes-p={}.p.split'.format(percentile)), 'wb'))
        pickle.dump(dict_aics, open(os.path.join(folder, 'dict_aics-p={}.p.split'.format(percentile)), 'wb'))
        pickle.dump(dict_ss, open(os.path.join(folder, 'dict_ss-p={}.p.split'.format(percentile)), 'wb'))

    print('done')
