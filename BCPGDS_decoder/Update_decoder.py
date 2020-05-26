from .Forward_Backward_augment_decoder import *
from .Config import *
from .GPU_Sampler import *
from .PGBN_sampler import *
#import Forward_augment
import scipy
import numpy as np
import time
import copy
from .Utils import *


def updatePhi_Pi(sweepi, X_train, Params, Data, SuperParams, MBt, Setting, W_left, W_right, epsit):
    MBObserved = (sweepi * Setting.batch_num + MBt).astype('int')
    train_doc_batch = Data.train_doc_split[MBt * Setting.batch_size:(MBt + 1) * Setting.batch_size]
    Batch_Sparse = Empty()
    Batch_Sparse.rows = []
    Batch_Sparse.cols = []
    Batch_Sparse.values = []
    Batch_Sparse.word2sen = []
    Batch_Sparse.word2doc = []
    Batch_Sparse.sen2doc = []
    Batch_Sparse.sen_len = []
    Batch_Sparse.doc_len = []
    for Doc_index, Doc in enumerate(train_doc_batch):

        for Sen_index, Sen in enumerate(Doc):
            Batch_Sparse.rows.extend(Sen)
            Batch_Sparse.cols.extend([i for i in range(len(Sen))])
            Batch_Sparse.values.extend([25 for i in range(len(Sen))])
            Batch_Sparse.word2sen.extend(
                [len(Batch_Sparse.sen_len) for i in range(len(Sen))])  # the sentence index for word
            Batch_Sparse.word2doc.extend([Doc_index for i in range(len(Sen))])
            Batch_Sparse.sen2doc.append(Doc_index)  # the document index for sentence
            Batch_Sparse.sen_len.append(len(Sen))  # the word number for each sentence

        Batch_Sparse.doc_len.append(len(Doc))  # the sentence number for each doc

    Batch_Sparse.max_doc_len = np.max(np.array(Batch_Sparse.doc_len))  # the max sentence number for each document

    # ======================= Setting CPGBN=======================#
    Setting.K1_V2 = np.max(np.array(Batch_Sparse.sen_len))  # the max word number for each sentence
    Setting.K1_S1 = Setting.K1_V1 - Setting.K1_S3 + 1
    Setting.K1_S2 = Setting.K1_V2 - Setting.K1_S4 + 1

    Setting.N_Sen = np.max(np.array(Batch_Sparse.word2sen)) + 1  # the number of total sentences

    # ======================= Initial Local Params =======================#
    # CPGBN
    Params.W1_nk1_left = np.random.rand(Setting.N_Sen, Setting.K1, Setting.K1_S1, Setting.K1_S2)
    Params.W1_nk1_right = np.random.rand(Setting.N_Sen, Setting.K1, Setting.K1_S1, Setting.K1_S2)
    Params.W1_nk1_left = W_left
    Params.W1_nk1_right = W_right
    Params.W1_nk1 = Params.W1_nk1_left + Params.W1_nk1_right  # N*K1*K1_S1*K1_S2

    # BPGDS
    Theta_knt_left = np.ones([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len])  # K1*Batch_size*T
    Theta_knt_right = np.ones([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len])  # K1*Batch_size*T
    Theta_knt = np.ones([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len])
    Zeta_nt_left = np.ones([Setting.batch_size, Batch_Sparse.max_doc_len + 1])  # Batch_size*T
    Zeta_nt_right = np.ones([Setting.batch_size, Batch_Sparse.max_doc_len + 1])  # Batch_size*T
    Delta_nt = np.ones([Setting.batch_size, Batch_Sparse.max_doc_len])  # Batch_size*T
    c2_nt = np.ones([Setting.batch_size, Batch_Sparse.max_doc_len])  # Batch_size*T

    # ===========================Collecting variables==================#

    # ======================= GPU Initial  =======================#
    X_rows = np.array(Batch_Sparse.rows, dtype=np.int32)  # rows
    X_cols = np.array(Batch_Sparse.cols, dtype=np.int32)  # cols
    X_values = np.array(Batch_Sparse.values, dtype=np.int32)
    X_sen_index = np.array(Batch_Sparse.word2sen, dtype=np.int32)  # pages

    word_total = len(X_rows)  # the number of word
    word_aug_stack = np.zeros((Setting.K1 * Setting.K1_S4 * word_total), dtype=np.float32)
    MultRate_stack = np.zeros((Setting.K1 * Setting.K1_S4 * word_total), dtype=np.float32)
    Batch_Para = np.array([Setting.K1, Setting.K1_S1, Setting.K1_S2, Setting.K1_S3, Setting.K1_S4, word_total],
                          dtype=np.int32)

    block_x = 128
    grid_x = 128
    grid_y = word_total / (block_x * grid_x) + 1

    time_Conv = 0

    # ====================== Augmentation ======================#
    Params.D1_k1_Aug = np.zeros_like(Params.D1_k1)
    Params.W1_nk1_Aug = np.zeros_like(Params.W1_nk1)

    W1_nk1 = np.array(Params.W1_nk1, dtype=np.float32, order='C')
    D1_k1 = np.array(Params.D1_k1, dtype=np.float32, order='C')
    W1_nk1_Aug = np.zeros(W1_nk1.shape, dtype=np.float32, order='C')
    D1_k1_Aug = np.zeros(D1_k1.shape, dtype=np.float32, order='C')

    time_1 = time.time()
    fuc = mod.get_function("Multi_Sampler")
    fuc(drv.In(Batch_Para), drv.In(word_aug_stack), drv.In(MultRate_stack), drv.In(X_rows), drv.In(X_cols),
            drv.In(X_sen_index),
            drv.In(X_values), drv.In(W1_nk1), drv.In(D1_k1), drv.InOut(W1_nk1_Aug), drv.InOut(D1_k1_Aug),
            grid=(int(grid_x), int(grid_y), 1), block=(int(block_x), 1, 1))
    time_2 = time.time()
    time_Conv += time_2 - time_1

    Params.W1_nk1_Aug = W1_nk1_Aug  # N*K1*K1_S1*K1_S2; Note: Don't add round here, case the scores are too small here!!!
    Params.D1_k1_Aug = D1_k1_Aug  # K1*K1_S3*K1_S4
    Params.W1_nk1_Aug_Pool = np.sum(np.sum(Params.W1_nk1_Aug, axis=3, keepdims=True), axis=2, keepdims=True)  # N*K1
    Params.W1_nk1_Aug_Rate = Params.W1_nk1_Aug / (Params.W1_nk1_Aug_Pool + real_min)  # N*K1*K1_S1*K1_S2

    # ====================== Augmentation ======================#

    # ======================separate forward and backward ===============#
    Params.W1_nk1_Aug_Pool_left = np.sum(Params.W1_nk1_left, axis=3, keepdims=True) / (
                    np.sum(Params.W1_nk1, axis=3, keepdims=True) + real_min) * Params.W1_nk1_Aug_Pool
    Params.W1_nk1_Aug_Pool_right = np.sum(Params.W1_nk1_right, axis=3, keepdims=True) / (
                    np.sum(Params.W1_nk1, axis=3, keepdims=True) + real_min) * Params.W1_nk1_Aug_Pool

    A_knt_left = np.zeros([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len])  # K1*Batch_size*T
    A_knt_right = np.zeros([Setting.K1, Setting.batch_size, Batch_Sparse.max_doc_len])

    for n in range(Setting.batch_size):
        A_sen_index = np.array(np.where(np.array(Batch_Sparse.sen2doc) == n))
        A_kt_left = np.transpose(Params.W1_nk1_Aug_Pool_left[A_sen_index[0, :], :, 0, 0])  # K1*T
        A_knt_left[:, n, -Batch_Sparse.doc_len[n]:] = A_kt_left  # K1*Batch_size*T
        A_kt_right = np.transpose(Params.W1_nk1_Aug_Pool[A_sen_index[0, :], :, 0, 0])  # K1*T
        A_knt_right[:, n, -Batch_Sparse.doc_len[n]:] = A_kt_right  # K1*Batch_size*T

    ##=============foward============##
    [Zeta_nt_left,  Z_kkdot_left] = \
            Forward_augment(Setting, SuperParams, Batch_Sparse, A_knt_left, Params, Theta_knt_left, Zeta_nt_left,
                            Delta_nt, c2_nt)
    ##=============backward============##
    [Zeta_nt_right,  Z_kkdot_right] = \
            Backward_augment(Setting, SuperParams, Batch_Sparse, A_knt_right, Params, Theta_knt_right,
                             Zeta_nt_right, Delta_nt, c2_nt)

    EWSZS_D = Params.D1_k1_Aug
    EWSZS_Pi_left =  Z_kkdot_left
    EWSZS_Pi_right = Z_kkdot_right

    Phi = np.transpose(np.reshape(Params.D1_k1, [Setting.K1, Setting.K1_S3 * Setting.K1_S4]))
    EWSZS_D = np.transpose(np.reshape(EWSZS_D, [Setting.K1, Setting.K1_S3 * Setting.K1_S4]))
    EWSZS_D = Setting.batch_num * EWSZS_D / Setting.Collection
    EWSZS_Pi_left = Setting.batch_num * EWSZS_Pi_left / Setting.Collection
    EWSZS_Pi_right = Setting.batch_num * EWSZS_Pi_right / Setting.Collection

    if (MBObserved == 0):
        NDot_D = EWSZS_D.sum(0)
        NDot_Pi_left = EWSZS_Pi_left.sum(0)
        NDot_Pi_right = EWSZS_Pi_left.sum(0)
    else:
        NDot_D = (1 - Setting.ForgetRate[MBObserved]) * NDot_D + Setting.ForgetRate[MBObserved] * EWSZS_D.sum(0)
        NDot_Pi_left = (1 - Setting.ForgetRate[MBObserved]) * NDot_Pi_left + Setting.ForgetRate[
            MBObserved] * EWSZS_Pi_left.sum(0)
        NDot_Pi_right = (1 - Setting.ForgetRate[MBObserved]) * NDot_Pi_right + Setting.ForgetRate[
            MBObserved] * EWSZS_Pi_right.sum(0)

    # Update D
    tmp = EWSZS_D + SuperParams.eta
    tmp = (1 / (NDot_D + real_min)) * (tmp - tmp.sum(0) * Phi)
    tmp1 = (2 / (NDot_D + real_min)) * Phi
    tmp = Phi + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Phi.shape[0],
                                                                                              Phi.shape[1])
    Phi = PGBN_sampler.ProjSimplexSpecial(tmp, Phi, 0)
    Params.D1_k1 = np.reshape(np.transpose(Phi), [Setting.K1, Setting.K1_S3, Setting.K1_S4])

    # Update Pi_left
    Pi_prior = np.eye(Setting.K1)
    # Pi_prior = np.dot(Params.V, np.transpose(Params.V))
    # Pi_prior[np.arange(K), np.arange(K)] = 0
    # Pi_prior = Pi_prior + np.diag(np.reshape(Params.Xi*Params.V, [K, 1]))

    tmp = EWSZS_Pi_left + Pi_prior
    tmp = (1 / (NDot_Pi_left + real_min)) * (tmp - tmp.sum(0) * Params.Pi_left)
    tmp1 = (2 / (NDot_Pi_left + real_min)) * Params.Pi_left
    tmp = Params.Pi_left + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(
        Params.Pi_left.shape[0], Params.Pi_left.shape[1])
    Params.Pi_left = PGBN_sampler.ProjSimplexSpecial(tmp, Params.Pi_left, 0)

    # Update Pi_right
    Pi_prior = np.eye(Setting.K1)
    # Pi_prior = np.dot(Params.V, np.transpose(Params.V))
    # Pi_prior[np.arange(K), np.arange(K)] = 0
    # Pi_prior = Pi_prior + np.diag(np.reshape(Params.Xi*Params.V, [K, 1]))

    tmp = EWSZS_Pi_right + Pi_prior
    tmp = (1 / (NDot_Pi_right + real_min)) * (tmp - tmp.sum(0) * Params.Pi_right)
    tmp1 = (2 / (NDot_Pi_right + real_min)) * Params.Pi_right
    tmp = Params.Pi_right + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(
        Params.Pi_right.shape[0], Params.Pi_right.shape[1])
    Params.Pi_right = PGBN_sampler.ProjSimplexSpecial(tmp, Params.Pi_right, 0)



    return Params.D1_k1, Params.Pi_left, Params.Pi_right
