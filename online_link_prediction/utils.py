import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import math
import pdb


def calc_ppr_by_power_iteration(P: sp.spmatrix, alpha: float, h: np.ndarray, t: int) -> np.ndarray:
    iterated = (1 - alpha) * h
    result = iterated.copy()
    for iteration in range(t):
        # iterated = (alpha * P).dot(iterated)
        iterated =  alpha * P @ iterated #alpha * P @ iterated
        result += iterated
    return result


def to_prmatrix(P: sp.spmatrix):
    sums = P.sum(axis = 0)
    Q = sp.lil_matrix(P.shape)
    P_t = P.transpose()
    for i in range(P.shape[0]):
        if sums[0, i] != 0:
            Q[i, :] = P_t[i, :]/sums[0, i]
    Q = Q.transpose()
    return Q.tocsr()


# udpate v from $\mathbf{v} =  \alpha \mathbf{A} \mathbf{v} + (1-\alpha) \mathbf{h}$ to $\mathbf{v} =  \alpha \mathbf{B} \mathbf{v} + (1-\alpha) \mathbf{h}$
def osp(v: np.ndarray, A: sp.spmatrix, B: sp.spmatrix, alpha: float, epsilon: float, whether_print: int) -> np.ndarray:
    assert A.shape == B.shape, "in osp, the dimension of matrix A should be the same as the dimension of matrix B"
    q_offset = alpha * (B - A) @ v
    v_offset = q_offset.copy()
    x_offset = q_offset.copy()
    number = 0
    # pdb.set_trace()
    while (np.linalg.norm(x_offset, 1) > epsilon):
        number += 1 
        x_offset = alpha * B @ x_offset
        v_offset += x_offset
    if whether_print == 1:
        # print("x_offset norm of osp:", np.linalg.norm(x_offset, 1), "iteration number of osp:", number)
        print(np.linalg.norm(x_offset, 1), number)
        pass
    return v + v_offset


# udpate v from a previous solution. The return value approximately satisfies $\mathbf{v} =  \alpha \mathbf{P} \mathbf{v} + (1-\alpha) \mathbf{h}$
def gauss_southwell(v: np.ndarray, P: sp.spmatrix, h: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    dimension_P = P.shape[0]
    x = v
    r = (1 - alpha) * h - (sp.eye(dimension_P) - alpha * P) @ v
    print("r is :", r)
    # print(f"r shape is : {r.shape}")
    # pdb.set_trace()
    max_index = np.argmax(r)
    number = 0
    print("r value is:\n", r)
    print("r max  value is:\n", r[max_index])
    while r[max_index] > epsilon:
        e = np.zeros((dimension_P,1))
        e[max_index] = 1
        x = x + r[max_index] * e
        # print(f"shape of e is {e.shape}")
        # print(f"r[max] is {r[max_index]}")
        # print(f"shape of P is {P.shape}")
        r = r - np.squeeze(r[max_index]) * e + alpha * np.squeeze(r[max_index]) * P @ e
        max_index = np.argmax(r)
        number += 1 
    print("number of iteration is:",number)
    # print("x shape is",x.shape)
    # print("final residual maximum element:", r[np.argmax(r)])
    return x



# udpate v from $\mathbf{v} =  \alpha \mathbf{P_old} \mathbf{v} + (1-\alpha) \mathbf{h_old}$ to $\mathbf{v} =  \alpha \mathbf{P_new} \mathbf{v} + (1-\alpha) \mathbf{h_new}$
def EvePPR_APP(v: np.ndarray, P_old: sp.spmatrix, P_new: sp.spmatrix, h_new:np.ndarray, alpha: float, epsilon: float, epsilon2: float):
    # v_mid = osp(v, P_old, P_new, alpha, epsilon, 1)
    return gauss_southwell(v, P_new, h_new, alpha, epsilon2)



def calc_onehot_ppr_matrix(P: sp.spmatrix, alpha: float, t: int) -> np.ndarray:
    iterated = (1 - alpha) * sp.eye(P.shape[0])
    matrix_result = iterated.copy()
    for iteration in range(t):
        iterated = (alpha * P).dot(iterated)
        matrix_result += iterated
    return matrix_result            