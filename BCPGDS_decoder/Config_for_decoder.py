from .Utils import *
import numpy as np
from .Read_IMDB import Load_Data



def decoder_setting(input_data, config):
    Setting = Empty()
    SuperParams = Empty()

    Params = Empty()
    Data = Empty()

    Data.word2index, Data.train_doc_split, Data.train_doc_label, Data.train_doc_len, \
    Data.test_doc_split, Data.test_doc_label, Data.test_doc_len = Load_Data(input_data)

    #def my_key(k):
        #return Data.train_doc_len[k]

    #sorted_train_index = sorted(range(len(Data.train_doc_len)), key=my_key)

    #def my_key(k):
        #return Data.test_doc_len[k]

    #sorted_test_index = sorted(range(len(Data.test_doc_len)), key=my_key)

    #Data.train_doc_split = [Data.train_doc_split[i] for i in sorted_train_index]
    #Data.train_doc_len = [Data.train_doc_len[i] for i in sorted_train_index]
    #Data.train_doc_label = [Data.train_doc_label[i] for i in sorted_train_index]
    #Data.test_doc_split = [Data.test_doc_split[i] for i in sorted_test_index]
    #Data.test_doc_len = [Data.test_doc_len[i] for i in sorted_test_index]
    #Data.test_doc_label = [Data.test_doc_label[i] for i in sorted_test_index]

    # ======================= Setting =======================#


    Setting.taoOFR = 0
    Setting.kappaOFR = 0.9
    Setting.kappa0 = 0.7
    Setting.tao0 = 20
    Setting.epsi0 = 1

    Setting.batch_size = config.batch_size
    Setting.Iter = 20
    Setting.Burnin = 0.6 * Setting.Iter
    Setting.Collection = Setting.Iter - Setting.Burnin

    Setting.K1 = config.decoder_channel

    # PGDS
    Setting.Station = 0
    Setting.N_Doc = len(Data.train_doc_split)
    Setting.N_Doc_te = len(Data.test_doc_split)

    Setting.batch_num = np.floor(Setting.N_Doc / Setting.batch_size).astype('int')
    Setting.iterall = Setting.SweepTime * Setting.batch_num
    Setting.ForgetRate = np.power((Setting.taoOFR + np.linspace(1, Setting.iterall, Setting.iterall)),
                                  -Setting.kappaOFR)
    epsit = np.power((Setting.tao0 + np.linspace(1, Setting.iterall, Setting.iterall)), -Setting.kappa0)
    epsit = Setting.epsi0 * epsit / epsit[0]

    Setting.K1_V1 = len(Data.word2index)
    Setting.K1_S3 = Setting.K1_V1
    Setting.K1_S4 = 3
    Setting.K1_S1 = Setting.K1_V1 - Setting.K1_S3 + 1

    # ======================= Initial Global Params =======================#


    # CPGBN
    # ======================= SuperParams =======================#


    # CPGBN
    SuperParams.gamma0 = 0.1  # r
    SuperParams.c0 = 0.1
    SuperParams.a0 = 0.1  # p
    SuperParams.b0 = 0.1
    SuperParams.e0 = 0.1  # c
    SuperParams.f0 = 0.1
    SuperParams.eta = 0.05  # Phi

    # PGDS
    SuperParams.tao0 = 1
    SuperParams.epilson0 = 0.1

    Params.D1_k1 = np.random.rand(Setting.K1, Setting.K1_S3, Setting.K1_S4)
    for k1 in range(Setting.K1):
        Params.D1_k1[k1, :, :] = Params.D1_k1[k1, :, :] / np.sum(Params.D1_k1[k1, :, :])

    # BPGDS

    Params.Pi_left = np.eye(Setting.K1)
    Params.Pi_right = np.eye(Setting.K1)
    Params.V_left = 0.1 * np.ones([Setting.K1, 1])
    Params.V_right = 0.1 * np.ones([Setting.K1, 1])

    # local
    Params.Theta_kt_all_left = [0] * Setting.N_Doc
    Params.Theta_kt_all_right = [0] * Setting.N_Doc
    Params.Zeta_t_all_left = [0] * Setting.N_Doc
    Params.Zeta_t_all_right = [0] * Setting.N_Doc
    Params.Delta_t_all = [0] * Setting.N_Doc


    # CPGBN

    Params.D1_k1 = np.random.rand(Setting.K1, Setting.K1_S3, Setting.K1_S4)
    for k1 in range(Setting.K1):
        Params.D1_k1[k1, :, :] = Params.D1_k1[k1, :, :] / np.sum(Params.D1_k1[k1, :, :])

    # BPGDS

    Params.Pi_left = np.eye(Setting.K1)
    Params.Pi_right = np.eye(Setting.K1)
    Params.V_left = 0.1 * np.ones([Setting.K1, 1])
    Params.V_right = 0.1 * np.ones([Setting.K1, 1])

    # local
    Params.Theta_kt_all_left = [0] * Setting.N_Doc
    Params.Theta_kt_all_right = [0] * Setting.N_Doc
    Params.Zeta_t_all_left = [0] * Setting.N_Doc
    Params.Zeta_t_all_right = [0] * Setting.N_Doc
    Params.Delta_t_all = [0] * Setting.N_Doc

    return Data, Setting, Params, SuperParams, epsit
