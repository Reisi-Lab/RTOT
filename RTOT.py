import numpy as np
import tensorly as tl
from tensorly.tenalg.proximal import soft_thresholding, svd_thresholding
from numpy.linalg import norm


def ttt(x, b, L, i=2):
    axis_x = []
    axis_b = []
    base = [*x.shape[1:]]
    length = 1 if i < L else L

    for i in range(length):
        if b.shape[i] in base:
            axis_x.append(base.index(b.shape[i]) + 1)
            axis_b.append(i)
            base[base.index(b.shape[i])] = None
    return np.tensordot(x, b, axes=(axis_x, axis_b))


def rtot(**params):
    x = params['x']
    y = params['y']
    dims = params['dims']
    L = params['L']
    M = params['M']
    Ru = params['Ru']
    m1 = params['m1']
    m2 = params['m2']
    m3 = params['m3']
    max_itr = params['max_itr'] if 'max_itr' in params else 100
    tol = params['tol'] if 'tol' in params else 1e-1

    assert L + M == len(dims)

    rns = np.random.RandomState()
    U = [rns.normal(size=(d, Ru)) for d in dims]
    J = [rns.normal(size=(d, Ru)) for d in dims]
    Z = [np.zeros_like(u) for u in U]
    s = np.zeros_like(y)
    W = [tl.cp_to_tensor((None, U))]
    Js = [tl.cp_to_tensor((None, J))]
    itr = 0

    for r in range(Ru):
        for i in range(len(dims)):
            U[i][:, r] = U[i][:, r] / norm(U[i][:, r])

    while True:
        for i in range(len(dims)):
            J[i] = svd_thresholding(U[i] + (1 / m3) * Z[i], 1 / m3)

            c = []
            for r in range(Ru):
                cr = []
                for u in [U[j] for j in range(len(U)) if j != i]:
                    cr.append(u[:, r].reshape(-1, 1))
                cr = tl.cp_to_tensor((None, cr))
                cr = ttt(x, cr, L, i)
                cr = tl.unfold(cr, 1).T if i < L else tl.tensor_to_vec(tl.unfold(cr, 1)).reshape(-1, 1)
                c.append(cr)
            c = np.concatenate(c, axis=1)

            if i < L:
                y_vec = tl.tensor_to_vec(y)
                s_vec = tl.tensor_to_vec(s)
                j_vec = tl.tensor_to_vec(J[i])
                z_vec = tl.tensor_to_vec(Z[i])
                ctc = c.T @ c
                u = tl.solve(m1 * ctc + m3 * np.eye(ctc.shape[0]), m1 * c.T @ (y_vec - s_vec) + m3 * j_vec - z_vec)
                U[i] = u.reshape(U[i].shape)
            else:
                y_mode = tl.unfold(y, i - L + 1)
                s_mode = tl.unfold(s, i - L + 1)
                ctc = c.T @ c
                u = tl.solve(m1 * ctc + m3 * np.eye(ctc.shape[0]), m1 * c.T @ (y_mode.T - s_mode.T) + m3 * J[i].T - Z[i].T)
                U[i] = u.T

        for i in range(len(dims)):
            Z[i] += m3 * (U[i] - J[i])

        W.append(tl.cp_to_tensor((None, U)))
        Js.append(tl.cp_to_tensor((None, J)))
        s = soft_thresholding(y - ttt(x, W[-1], L), m2 / m1)

        norm_s = tl.norm(Js[-2] - Js[-1], 2) ** 2
        norm_r = tl.norm(W[-1] - Js[-1]) ** 2

        if norm_r > 10 * norm_s:
            m3 *= 2
        elif norm_s > 10 * norm_r:
            m3 /= 2

        if (norm_r < tol and norm_s < tol) or (itr > max_itr):
            print('norm_r: {:.5f}, norm_s: {:.5f}, itr: {}'.format(norm_r, norm_s, itr))
            return W[-1], s # dict(b_pre=W[-1], s_pre=s)
        itr = itr + 1


if __name__ == '__main__':
    params = dict(
        x=np.random.random(size=(16, 3, 15)),
        y=np.random.random(size=(16, 8, 10)),
        dims=(3, 15, 8, 10),
        L=2,
        M=2,
        Ru=3,
        m1=.4,
        m2=.5,
        m3=3e-4,
        tol=1e-3,
        max_itr=1000
    )
    weights, s = rtot(**params)
    # print(weights.shape)
    # print('done')
