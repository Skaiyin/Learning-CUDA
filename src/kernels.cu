#include <vector>

#include "../tester/utils.h"

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
__global__ void partition_and_count_kernel(
    T* data, T* temp, size_t n, T pivot, 
    int* counts, int* less_idx, int* greater_idx) 
{
    // 每个线程处理多个元素
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // 第一阶段：原子计数
    for (size_t i = tid; i < n; i += stride) {
        if (data[i] < pivot) {
            atomicAdd(&counts[0], 1);
        } else if (data[i] == pivot) {
            atomicAdd(&counts[1], 1);
        } else {
            atomicAdd(&counts[2], 1);
        }
    }
    __syncthreads();

    // 第二阶段：实际分区
    for (size_t i = tid; i < n; i += stride) {
        if (data[i] < pivot) {
            int pos = atomicAdd(less_idx, 1);
            temp[pos] = data[i];
        } else if (data[i] > pivot) {
            int pos = atomicAdd(greater_idx, 1);
            temp[n - 1 - pos] = data[i]; // 反向存储
        }
    }
}

template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
    // 初始化检查
    if (k == 0 || k > h_input.size()) return T(-100);

    // 设备内存分配
    T *d_data, *d_temp;
    int *d_counts, *d_less_idx, *d_greater_idx;
    cudaMalloc(&d_data, h_input.size() * sizeof(T));
    cudaMalloc(&d_temp, h_input.size() * sizeof(T));
    cudaMalloc(&d_counts, 3 * sizeof(int));
    cudaMalloc(&d_less_idx, sizeof(int));
    cudaMalloc(&d_greater_idx, sizeof(int));

    cudaMemcpy(d_data, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);

    size_t left = 0;
    size_t right = h_input.size();
    size_t target_k = k;
    T result = T(-100);

    const int blockSize = 256;

    while (left < right) {
        // 1. 选择pivot（改进为随机选择更好）
        T pivot;
        cudaMemcpy(&pivot, &d_data[left + (right - left)/2], sizeof(T), cudaMemcpyDeviceToHost);

        // 2. 初始化计数和索引
        int init_counts[3] = {0, 0, 0};
        cudaMemcpy(d_counts, init_counts, 3 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_less_idx, 0, sizeof(int));
        cudaMemset(d_greater_idx, 0, sizeof(int));

        // 3. 执行分区kernel
        int gridSize = (right - left + blockSize - 1) / blockSize;
        partition_and_count_kernel<<<gridSize, blockSize>>>(
            d_data + left, d_temp, right - left, pivot, 
            d_counts, d_less_idx, d_greater_idx);
        cudaDeviceSynchronize();

        // 4. 获取分区结果
        int counts[3];
        cudaMemcpy(counts, d_counts, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        int less = counts[0], equal = counts[1], greater = counts[2];

        // 5. 重组数据（将分区结果写回原数组）
        // less部分已经在temp[0..less-1]
        // equal部分需要从原数组复制
        // greater部分在temp[n-greater..n-1]
        cudaMemcpy(d_data + left, d_temp, less * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_data + left + less + equal, d_temp + (right-left) - greater, 
                  greater * sizeof(T), cudaMemcpyDeviceToDevice);

        // 6. 决定搜索范围
        if (target_k <= greater) {
            left = left + less + equal;
        } else if (target_k <= greater + equal) {
            result = pivot;
            break;
        } else {
            right = left + less;
            target_k -= (greater + equal);
        }
    }

    // 资源清理
    cudaFree(d_data);
    cudaFree(d_temp);
    cudaFree(d_counts);
    cudaFree(d_less_idx);
    cudaFree(d_greater_idx);

    return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
