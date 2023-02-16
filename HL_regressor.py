import math

import torch
import tensorly as tl
import numpy as np
import HL_data_io as io
import pandas as pd
import plotly.express as px

from tensorly.tenalg.proximal import soft_thresholding, svd_thresholding
from tensorly.decomposition import robust_pca
from numpy.linalg import norm


def ttt(x, b, L, dims):
    return np.tensordot(
        x,
        b,
        axes=[
            [k + 1 for k in range(len(dims[:L]))],
            [k for k in range(len(dims[:L]))]
        ]
    )


def aic(y_true, y_pre, s_pre, r, b):
    # print(np.log(norm(y_true - y_pre - s_pre) ** 2))
    # print(np.count_nonzero(s_pre), s_pre.size, np.log(np.count_nonzero(s_pre)))
    # print(.5 * r)
    return np.log(norm(y_true - y_pre - s_pre) ** 2) + 2 * np.log(np.count_nonzero(s_pre)) + .5 * r


class TOT:
    def __init__(self, **params):
        self.dims = params['dims']
        self.idx = [i for i in range(len(self.dims))]
        self.L = params['L']
        self.M = params['M']
        self.N = params['N']
        self.R = params['R']
        self.max_itr = params['max_itr']
        self.x = params['x']
        self.y = params['y']
        self.y_vec = tl.tensor_to_vec(params['y'])
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]
        self.y_vec_cuda = torch.tensor(self.y_vec, device='cuda')
        self.yms_cuda = [torch.tensor(ym, device='cuda') for ym in self.yms]
        self.name = 'tot'

    def fit(self, verbose=False):
        results = []
        U = [np.random.rand(p, self.R) for p in self.dims]
        itr = 0

        if verbose:
            print('======================')
            print('         TOT          ')
            print('======================')

        while itr < self.max_itr:
            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))
                    cr = np.tensordot(
                        self.x,
                        cr,
                        axes=[
                            [k + 1 for k in range(len(self.dims[:self.L])) if k != i],
                            [k for k in [l for l in range(len(cr.shape))][:-self.M]]
                        ]
                    )
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)
                C = np.concatenate(C)
                C = torch.tensor(C, device='cuda')
                u = (torch.pinverse(C.T) @ self.y_vec_cuda).reshape(-1, self.R)
                U[i] = u.cpu().numpy()
                # U[i] = (np.linalg.pinv(C.T) @ tl.tensor_to_vec(self.y)).reshape(-1, self.R)

            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))
                D = np.concatenate(D, axis=1)
                D = torch.tensor(D, device='cuda')
                U[self.L + i] = (torch.pinverse(D) @ self.yms_cuda[i + 1].T).T.cpu().numpy()
                # U[self.L + i] = (np.linalg.pinv(D) @ tl.unfold(self.y, mode=i + 1).T).T

            B = tl.cp_to_tensor((None, U))
            y_pre = ttt(self.x, B, self.L, self.dims)
            rpe = (np.linalg.norm(self.y - y_pre)) / np.linalg.norm(self.y)

            if len(results) > 1 and abs(rpe - results[-1][0]) < 1e-4:
                break
            if verbose:
                msg = 'itr={}, rpe={:.4f}'.format(itr + 1, rpe)
                print(msg)

            results.append((rpe, B, 0, 0))
            itr += 1

        if verbose:
            print('======================')
        results = sorted(results, key=lambda x: x[0])
        return results[-1][0], results[-1][1], results[-1][2], results[-1][3]


class RTOT:
    def __init__(self, **params):
        self.dims = params['dims']
        self.idx = [i for i in range(len(self.dims))]
        self.L = params['L']
        self.M = params['M']
        self.N = params['N']
        self.R = params['R']
        self.max_itr = params['max_itr']
        self.x = params['x']
        self.y = params['y']
        self.mu1 = params['mu1']
        self.mu2 = params['mu2']
        self.mu3 = params['mu3']
        self.tol = params['tol']
        self.y_vec = tl.tensor_to_vec(params['y'])
        self.yms = [tl.unfold(self.y, i) for i in range(len(self.y.shape))]
        self.y_vec_cuda = torch.tensor(self.y_vec, device='cuda')
        self.yms_cuda = [torch.tensor(ym, device='cuda') for ym in self.yms]
        self.name = 'rtot'

    def fit(self, verbose=False):
        U = [np.random.rand(p, self.R) for p in self.dims]
        J = [np.random.rand(p, self.R) for p in self.dims]
        Z = [np.zeros_like(u) for u in U]
        S = np.zeros_like(self.y)
        results = []
        Bs = [tl.cp_to_tensor((None, U))]
        Js = [tl.cp_to_tensor((None, J))]
        itr = 0

        if verbose:
            print('======================')
            print('        RTOT          ')
            print('======================')

        while itr < self.max_itr:
            for i in range(len(J)):
                J[i] = svd_thresholding(U[i] + (1 / self.mu3) * Z[i], 1 / self.mu3)

            for i in range(len(self.dims[:self.L])):
                C = []
                for r in range(self.R):
                    cr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != i]
                    cr = tl.cp_to_tensor((None, cr))
                    cr = np.tensordot(
                        self.x,
                        cr,
                        axes=[
                            [k + 1 for k in range(len(self.dims[:self.L])) if k != i],
                            [k for k in [l for l in range(len(cr.shape))][:-self.M]]
                        ]
                    )
                    cr = tl.unfold(cr, mode=1)
                    C.append(cr)
                C = np.concatenate(C)
                C = torch.tensor(C, device='cuda')
                CCT = C @ C.T
                s_vec = torch.tensor(S, device='cuda').flatten()
                j_vec = torch.tensor(J[i], device='cuda').flatten()
                z_vec = torch.tensor(Z[i], device='cuda').flatten()
                U[i] = (torch.pinverse(self.mu1 * CCT + self.mu3 * torch.eye(n=CCT.shape[0], device='cuda')) @ (self.mu1 * C @ (self.y_vec_cuda - s_vec) + self.mu3 * j_vec - z_vec)).reshape(-1,
                                                                                                                                                                                              self.R).cpu().numpy()
                # y_vec = tl.tensor_to_vec(self.y)
                # s_vec = tl.tensor_to_vec(S)
                # j_vec = tl.tensor_to_vec(J[i])
                # z_vec = tl.tensor_to_vec(Z[i])
                # CCT = C @ C.T
                # U[i] = (np.linalg.pinv(self.mu1 * CCT + self.mu3 * np.eye(CCT.shape[0])) @ (self.mu1 * C @ (y_vec - s_vec) + self.mu3 * j_vec - z_vec)).reshape(-1, self.R)

            for i in range(len(self.dims[self.L:])):
                D = []
                for r in range(self.R):
                    dr = [U[j][:, r].reshape(-1, 1) for j in range(len(U)) if j != self.L + i]
                    dr = tl.cp_to_tensor((None, dr))
                    dr = ttt(self.x, dr, self.L, self.dims)
                    D.append(tl.tensor_to_vec(dr).reshape(-1, 1))
                D = np.concatenate(D, axis=1)
                D = torch.tensor(D, device='cuda')
                DTD = D.T @ D
                ym = self.yms_cuda[i + 1].T
                sm = torch.tensor(tl.unfold(S, mode=i + 1).T, device='cuda')
                J_cuda = torch.tensor(J[self.L + i], device='cuda')
                Z_cuda = torch.tensor(Z[self.L + i], device='cuda')
                U[self.L + i] = (torch.pinverse(self.mu1 * DTD + self.mu3 * torch.eye(n=DTD.shape[0], device='cuda')) @ (self.mu1 * D.T @ (ym - sm) + self.mu3 * J_cuda.T - Z_cuda.T)).T.cpu().numpy()
                # DTD = D.T @ D
                # ym = tl.unfold(self.y, mode=i + 1).T
                # sm = tl.unfold(S, mode=i + 1).T
                # U[self.L + i] = (np.linalg.pinv(self.mu1 * DTD + self.mu3 * np.eye(DTD.shape[0])) @ (self.mu1 * D.T @ (ym - sm) + self.mu3 * J[self.L + i].T - Z[self.L + i].T)).T

            for i in range(len(Z)):
                Z[i] += self.mu3 * (U[i] - J[i])

            B = tl.cp_to_tensor((None, U))
            J_ = tl.cp_to_tensor((None, J))
            y_pre = ttt(self.x, B, self.L, self.dims)
            S = soft_thresholding(self.y - y_pre, self.mu2 / self.mu1)
            rpe = (np.linalg.norm(self.y - y_pre)) / np.linalg.norm(self.y)
            norm_s = tl.norm(Js[-1] - J_, 2) ** 2
            norm_r = tl.norm(Bs[-1] - B, 2) ** 2
            sparsity = (S.size - np.count_nonzero(S)) / S.size
            AIC = aic(self.y, y_pre, S, self.R, B)

            # need to uncomment when normal use
            # if norm_r > 2 * norm_s:
            #     self.mu3 *= 2
            # elif norm_s > 2 * norm_r:
            #     self.mu3 /= 2

            if verbose:
                msg = 'itr={}, rpe={:.4f}, norm_s={:.4f}, norm_r={:.4f}, sparsity={:.2f}'.format(itr + 1, rpe, norm_s, norm_r, sparsity)
                print(msg)

            if len(results) > 1 and abs(rpe - results[-1][0]) < 1e-4:
                break
            if itr > self.max_itr or (norm_r < self.tol and norm_s < self.tol):
                break
            if norm_r < math.inf:
                results.append((rpe, B, norm_r, AIC, S))

            Bs.append(B)
            Js.append(J_)

            itr += 1

        if verbose:
            print('======================')

        # results = sorted(results, key=lambda x: x[0])
        return results[-1][0], results[-1][1], results[-1][3], results[-1][4]


class RPCA:
    def __init__(self, **params):
        self.params = params
        self.reg = params['reg'] if 'reg' in params else 1.5e-1
        self.name = 'rpca'

    def fit(self, verbose=False):
        if verbose:
            print('======================')
            print('        RPCA          ')
            print('======================')

        y, S = robust_pca(self.params['y'], reg_E=self.reg, verbose=verbose)

        if verbose:
            print('sparsity={:.2f}'.format((S.size - np.count_nonzero(S)) / S.size))

        self.params['y'] = y
        tot = TOT(**self.params)
        return tot.fit(verbose=verbose)


if __name__ == '__main__':
    dims = (15, 20, 5, 10)
    N = 30
    L = 2
    M = 2
    params = dict(
        # dims=(15, 20, 5, 10),
        # idx=[i for i in range(len(dims))],
        # L=2,
        # M=2,
        # N=30,
        R=3,
        mu1=5.5e-3,
        mu2=1e-3,
        mu3=1e-4,
        tol=1e-4,
        max_itr=100,
        density=.2,
        # x=np.random.rand(N, *dims[:L]),
        # y=np.random.rand(N, *dims[L:])
    )
    params = io.gen_lambda_data(**params)

    tot = TOT(**params)
    rtot = RTOT(**params)
    rpca = RPCA(**params)

    # _, B_rtot = rtot.fit(verbose=True)
    # _, B_tot = tot.fit(verbose=True)
    _, B_rpca = rpca.fit(verbose=True)

    y_true_list = []
    y_pre_list = []
    for x, y in zip(params['x_test'], params['y_test']):
        dims = params['dims']
        x = x.reshape(1, dims[0], dims[1])
        y_pre = ttt(x, B_rpca, params['L'], params['dims'])
        y_pre_list.append(y_pre.flatten())
        y_true_list.append(y.flatten())
    y_true_list, y_pre_list = np.array(y_true_list).T, np.array(y_pre_list).T
    y_pre_list = y_pre_list - np.mean(y_pre_list, axis=0) + np.mean(y_true_list, axis=0)
    data = np.concatenate((y_true_list, y_pre_list), axis=0)
    data = pd.DataFrame(data=data, columns=[i for i in range(1, data.shape[1] + 1)])
    data['time'] = [i for i in range(1, 204)] * 2
    data['type'] = ['true' for _ in range(203)] + ['pre' for _ in range(203)]

    for i in range(1, 10):
        fig = px.line(data, x='time', y=i, line_group='type', color='type')
        fig.show()
