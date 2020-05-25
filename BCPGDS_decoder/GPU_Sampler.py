# ====================== CUDA Initial ======================#
# Note， do not add any cuda operation among CUDA initial such as Tensorflow!!!!!!!!!!!!!!!!!!
import pycuda.curandom as curandom
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""

#include <stdio.h>

__device__ int cudarand(long long seed)
{
    long long temp=(48271 * seed + 0) % 2147483647;
    return temp;
}

/*__global__ void Multi_Sample_2(int* Para, int* X_value, int* X_rows, int* X_cols, float* Phi, float* Theta, float* XVK, float* XKJ)    //
{
    const int V = Para[0];
    const int K = Para[1];
    const int J = Para[2];
    const int N = Para[3];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        int v = X_rows[idx];
        int j = X_cols[idx];
        float sum=0.0;

        for (int k = 0; k < K; k++)
        {
            sum += Phi[v*K + k] * Theta[k*J + j];
        }

        for (int k = 0; k<K; k++)
        {
            float cumsum = Phi[v*K + k] * Theta[k*J + j] / sum;
            
            atomicAdd(&XVK[v*K + k], cumsum);
            atomicAdd(&XKJ[k*J + j], cumsum);
        }
        
    }
} */

__global__ void Multi_Sample(float* randomseed, int* Para, int* X_value, int* X_rows, int* X_cols, float* Phi, float* Theta, float* XVK, float* XKJ)    //
{
    const int V = Para[0];
    const int K = Para[1];
    const int J = Para[2];
    const int N = Para[3];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = randomseed[idx]*2147483647.0;

    if (idx < N)
    {
        float cumsum = 0.0;
        int v = X_rows[idx];
        int j = X_cols[idx];
        float sum=0.0;

        for (int k = 0; k < K; k++)
        {
            sum += Phi[v*K + k] * Theta[k*J + j];
        }
        
        for (int token = 0; token<X_value[idx]; token++)
        {
            seed = cudarand(seed);
            float probrnd = ((double)(seed) / 2147483647.0) * sum;
            int Embedding_K=K-1;
            float sumprob=0.0;
            for (int k = 0; k < K; k++)
            {
                cumsum = Phi[v*K + k] * Theta[k*J + j];
                if (sumprob+cumsum>=probrnd)
                {
                    Embedding_K=k;
                    break;
                }
                sumprob+=cumsum;
            }
            atomicAdd(&XVK[v*K + Embedding_K], 1);
            atomicAdd(&XKJ[Embedding_K*J + j], 1);
        }
    }
}

__global__ void Multi_Sampler(int* para, float* word_aug_stack, float* MultRate_stack, int* row_index, int* column_index, int* page_index, int* value_index, float* Params_W1_nk1, float* Params_D1_k1, float* Params_W1_nk1_Aug, float* Params_D1_k1_Aug)
{
    int K1         = para[0];
    int K1_K1      = para[1];
    int K1_K2      = para[2];
    int K1_K3      = para[3];
    int K1_K4      = para[4];
    int word_total = para[5];

    int ix = blockDim.x * blockIdx.x + threadIdx.x; 
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy* blockDim.x *gridDim.x+ ix;

    if ((idx < word_total))
    {
        int v1 = row_index[idx];                 // row_index
        int v2 = column_index[idx];              // col_index
        int n  = page_index[idx];                // file_index
        int value = value_index[idx];

        int word_k1_min = 0;
        int word_k1_max = 0;
        int word_k2_min = 0;
        int word_k2_max = 0;

        // word_k1
        if ((v1 - K1_K3 + 1) > 0)
            word_k1_min = v1 - K1_K3 + 1;
        else
            word_k1_min = 0;

        if (v1 > K1_K1 -1)
            word_k1_max = K1_K1 -1;
        else
            word_k1_max = v1;

        int l_word_k1 = word_k1_max - word_k1_min + 1;
        int *word_k1  = new int[l_word_k1];
        for (int i = 0; i < (l_word_k1); i++)
            word_k1[i] = word_k1_min + i;

        // word_k2
        if ((v2 - K1_K4 + 1) > 0)
            word_k2_min = v2 - K1_K4 + 1;
        else
            word_k2_min = 0;

        if (v2 > K1_K2 -1)
            word_k2_max = K1_K2 -1;
        else
            word_k2_max = v2;

        int l_word_k2 = word_k2_max - word_k2_min + 1;
        int *word_k2  = new int[l_word_k2];
        for (int i = 0; i < (l_word_k2); i++)
            word_k2[i] = word_k2_min + i;

        // word_k3
        int *word_k3 = new int[l_word_k1];
        for (int i = 0; i < (l_word_k1); i++)
            word_k3[i] = v1 - word_k1[i] ;

        // word_k4
        int *word_k4 = new int[l_word_k2];
        for (int i = 0; i < (l_word_k2); i++)
            word_k4[i] = v2 - word_k2[i] ;

        float MultRate_sum = 0;
        //word_aug_stack
        //MultRate_stack
        //Params_W1_nk1
        //Params_D1_k1
        int stack_start = idx * K1_K4 * K1;

        for (int i = 0; i < K1; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int temp_a = (n) * K1 * K1_K1 * K1_K2 + (i) * K1_K1 * K1_K2 + word_k1[k] * K1_K2 + (word_k2[j]);
                    int temp_b = (i) * K1_K3 * K1_K4 + word_k3[k] * K1_K4 + (word_k4[j]);
                    int temp_c = stack_start + i*l_word_k1*l_word_k2 + k*l_word_k2 + j;

                    MultRate_stack[temp_c] = Params_W1_nk1[temp_a] * Params_D1_k1[temp_b];
                    MultRate_sum = MultRate_sum + MultRate_stack[temp_c];
                }
            }
        }

        for (int i = 0; i < K1; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int temp_a = (n) * K1 * K1_K1 * K1_K2 + (i) * K1_K1 * K1_K2 + word_k1[k] * K1_K2 + (word_k2[j]);
                    int temp_b = (i) * K1_K3 * K1_K4 + word_k3[k] * K1_K4 + (word_k4[j]);
                    int temp_c = stack_start + i*l_word_k1*l_word_k2 + k*l_word_k2 + j;

                    if (MultRate_sum == 0)
                    {
                        MultRate_stack[temp_c] = 1.0 / (K1 * l_word_k1 * l_word_k2);
                        word_aug_stack[temp_c] = MultRate_stack[temp_c] * value;
                    }
                    else
                    {
                        MultRate_stack[temp_c] = MultRate_stack[temp_c] / MultRate_sum;
                        word_aug_stack[temp_c] = MultRate_stack[temp_c] * float(value);
                    }

                    atomicAdd(&Params_W1_nk1_Aug[temp_a], word_aug_stack[temp_c]);
                    atomicAdd(&Params_D1_k1_Aug[temp_b], word_aug_stack[temp_c]);
                }
            }
        }

        delete[] word_k1;
        delete[] word_k2;
        delete[] word_k3;
        delete[] word_k4; 
    }

}
 """)

def Multrnd_Matrix_GPU(X_t, Phi_t, Theta_t):

    func = mod.get_function('Multi_Sample')
    [V, J] = X_t.shape
    K = Theta_t.shape[0]

    # [X_t_rows, X_t_cols] = np.where(np.ones_like(X_t))
    # X_t_values = X_t[(X_t_rows, X_t_cols)]
    [X_t_rows, X_t_cols] = np.where(X_t > 0.5)
    X_t_values = X_t[(X_t_rows, X_t_cols)]

    N = len(X_t_values)  # number of sample point

    Para = np.array([V, K, J, N], dtype=np.int32)

    X_t_values = np.array(X_t_values, dtype=np.int32)
    X_t_rows = np.array(X_t_rows, dtype=np.int32)
    X_t_cols = np.array(X_t_cols, dtype=np.int32)

    Xt_to_t1_t = np.zeros([K, J], dtype=np.float32, order='C')
    WSZS_t = np.zeros([V, K], dtype=np.float32, order='C')
    Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
    Theta_t = np.array(Theta_t, dtype=np.float32, order='C')

    if N != 0:

        block_x = int(400)
        grid_x = int(np.floor(N / block_x) + 1)
        # print("block: {:<8.4f}, grid:{:<8.4f}".format(block_x, grid_x))

        randomseed = np.random.rand(N)
        randomseed = np.array(randomseed, dtype=np.float32, order='C')

        func(drv.In(randomseed), drv.In(Para), drv.In(X_t_values), drv.In(X_t_rows), drv.In(X_t_cols), drv.In(Phi_t),
             drv.In(Theta_t), drv.InOut(WSZS_t), drv.InOut(Xt_to_t1_t),
             grid=(grid_x, 1, 1), block=(block_x, 1, 1))

    return Xt_to_t1_t, WSZS_t


# def Multrnd_Matrix_GPU_2(X_t, Phi_t, Theta_t):
#
#     func = mod.get_function('Multi_Sample_2')
#     [V, J] = X_t.shape
#     K = Theta_t.shape[0]
#
#     # [X_t_rows, X_t_cols] = np.where(np.ones_like(X_t))
#     # X_t_values = X_t[(X_t_rows, X_t_cols)]
#     [X_t_rows, X_t_cols] = np.where(X_t > 0.5)
#     X_t_values = X_t[(X_t_rows, X_t_cols)]
#
#     N = len(X_t_values)  # number of sample point
#
#     Para = np.array([V, K, J, N], dtype=np.int32)
#
#     X_t_values = np.array(X_t_values, dtype=np.int32)
#     X_t_rows = np.array(X_t_rows, dtype=np.int32)
#     X_t_cols = np.array(X_t_cols, dtype=np.int32)
#
#     Xt_to_t1_t = np.zeros([K, J], dtype=np.float32, order='C')
#     WSZS_t = np.zeros([V, K], dtype=np.float32, order='C')
#     Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
#     Theta_t = np.array(Theta_t, dtype=np.float32, order='C')
#
#     if N != 0:
#
#         block_x = int(400)
#         grid_x = int(np.floor(N / block_x) + 1)
#         # print("block: {:<8.4f}, grid:{:<8.4f}".format(block_x, grid_x))
#
#         func(drv.In(Para), drv.In(X_t_values), drv.In(X_t_rows), drv.In(X_t_cols), drv.In(Phi_t),
#              drv.In(Theta_t), drv.InOut(WSZS_t), drv.InOut(Xt_to_t1_t),
#              grid=(grid_x, 1, 1), block=(block_x, 1, 1))
#
#     return Xt_to_t1_t, WSZS_t


print("CUDA initial finish")