from .Load_Data import *
from .Utils import *
from .Config import *
from .GPU_Sampler import *
from .PGBN_sampler import *
import scipy
import numpy as np
import time
import copy

def Forward_augment(Setting, SuperParams, Batch_Sparse, A_knt_left, Params, Theta_knt_left, Zeta_nt_left, Delta_nt,c2_nt):
    ##=============foward============##
    Z_kdotnt = np.zeros([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len + 1])  # K1*Batch_size*(T+1) for Augmented Matrix
    Z_dotknt = np.zeros([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len + 1])  # K1*Batch_size*(T+1) for Theta
    Z_kkdot_left = np.zeros([Setting.K1, Setting.K1])  # K1*K1 for Pi

    for t in range(Batch_Sparse.max_doc_len - 1, 0, -1):  # T-1 : 1　Augment T-1 times

        # the augmented input for the Theta t
        Q_kn = A_knt_left[:, :, t] + Z_dotknt[:, :, t + 1]  # K*N
        # the augmented input from layer t to layer t-1
        Z_kdotnt[:, :, t] = PGBN_sampler.Crt_Matrix(Q_kn.astype('double'),
                                                    SuperParams.tao0 * np.dot(Params.Pi_left, Theta_knt_left[:, :, t - 1]))
        # the augmented input from the Theta t-1
        # augmented with CPU
        # [Z_dotknt[:, :, t], Z_kkt] = PGBN_sampler.Multrnd_Matrix(np.array(Z_kdotnt[:, :, t], dtype=np.double, order='C'),
        #                                                          Params.Pi,
        #                                                          np.array(Theta_knt[:, :, t-1], dtype=np.double, order='C'))
        # augmented with GPU
        [Z_dotknt[:, :, t], Z_kkt] = Multrnd_Matrix_GPU(np.array(Z_kdotnt[:, :, t], dtype=np.double, order='C'),
                                                        Params.Pi_left,
                                                        np.array(Theta_knt_left[:, :, t - 1], dtype=np.double,
                                                                 order='C'))
        Z_kkdot_left = Z_kkdot_left + Z_kkt


    # Calculate Zeta
    if Setting.Station == 0:
        for t in range(Batch_Sparse.max_doc_len - 1, -1, -1):
            Zeta_nt_left[:, t] = np.log(1 + Zeta_nt_left[:, t + 1] + Delta_nt[:, t] / SuperParams.tao0)

    time_1 = time.time()
    return Zeta_nt_left, Delta_nt, c2_nt, Z_kkdot_left


def Backward_augment(Setting, SuperParams, Batch_Sparse, A_knt_right, Params, Theta_knt_right, Zeta_nt_right, Delta_nt,c2_nt):
    Z_kdotnt = np.zeros([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len + 1])  # K1*Batch_size*(T+1) for Augmented Matrix
    Z_dotknt = np.zeros([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len + 1])  # K1*Batch_size*(T+1) for Theta
    Z_kkdot_right = np.zeros([Setting.K1, Setting.K1])  # K1*K1 for Pi
    T = Batch_Sparse.max_doc_len

    for t in range(0, Batch_Sparse.max_doc_len - 1, 1):  # T-1 : 1　Augment T-1 times

        # the augmented input for the Theta t
        Q_kn = A_knt_right[:, :, t] + Z_dotknt[:, :, T-t]  # K*N
        # the augmented input from layer t to layer t-1
        Z_kdotnt[:, :, T-t-1] = PGBN_sampler.Crt_Matrix(Q_kn.astype('double'),
                                                    SuperParams.tao0 * np.dot(Params.Pi_right,
                                                                              Theta_knt_right[:, :,
                                                                              t + 1]))
        # the augmented input from the Theta t-1
        # augmented with CPU
        # [Z_dotknt[:, :, t], Z_kkt] = PGBN_sampler.Multrnd_Matrix(np.array(Z_kdotnt[:, :, t], dtype=np.double, order='C'),
        #                                                          Params.Pi,
        #                                                          np.array(Theta_knt[:, :, t-1], dtype=np.double, order='C'))
        # augmented with GPU
        [Z_dotknt[:, :, T-t], Z_kkt] = Multrnd_Matrix_GPU(
            np.array(Z_kdotnt[:, :, t], dtype=np.double, order='C'),
            Params.Pi_right,
            np.array(Theta_knt_right[:, :, t + 1], dtype=np.double, order='C'))
        Z_kkdot_right = Z_kkdot_right + Z_kkt

    # Calculate Zeta
    if Setting.Station == 0:
        for t in range(Batch_Sparse.max_doc_len):
            Zeta_nt_right[:, T-t] = np.log(1 + Zeta_nt_right[:, T - t] + Delta_nt[:, t] / SuperParams.tao0)

    return Zeta_nt_right, Delta_nt, c2_nt, Z_kkdot_right