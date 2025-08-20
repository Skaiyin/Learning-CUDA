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
template<typename T>
__global__ void computeAttentionScores(
    const T* q, const T* k, T* scores,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    T scale, bool is_causal) {
    
    int batch = blockIdx.x;
    int q_head = blockIdx.y;
    int i = blockIdx.z;
    int j = threadIdx.x;
    
    if (batch >= batch_size || q_head >= query_heads || 
        i >= target_seq_len || j >= src_seq_len) return;
    
    int head_group_size = query_heads / kv_heads;
    int kv_head = q_head / head_group_size;
    
    T score = static_cast<T>(0);
    for (int d = 0; d < head_dim; ++d) {
        int q_idx = batch * target_seq_len * query_heads * head_dim + 
                   i * query_heads * head_dim + q_head * head_dim + d;
        int k_idx = batch * src_seq_len * kv_heads * head_dim + 
                   j * kv_heads * head_dim + kv_head * head_dim + d;
        
        score += q[q_idx] * k[k_idx];
    }
    
    score *= scale;
    
    if (is_causal && j > i) {
        score = -static_cast<T>(1e30);
    }
    
    int score_idx = batch * query_heads * target_seq_len * src_seq_len +
                   q_head * target_seq_len * src_seq_len +
                   i * src_seq_len + j;
    scores[score_idx] = score;
    
    // Handle case where we have more threads than src_seq_len
    for (int jj = j + blockDim.x; jj < src_seq_len; jj += blockDim.x) {
        T score_extra = static_cast<T>(0);
        for (int d = 0; d < head_dim; ++d) {
            int q_idx = batch * target_seq_len * query_heads * head_dim + 
                       i * query_heads * head_dim + q_head * head_dim + d;
            int k_idx = batch * src_seq_len * kv_heads * head_dim + 
                       jj * kv_heads * head_dim + kv_head * head_dim + d;
            
            score_extra += q[q_idx] * k[k_idx];
        }
        
        score_extra *= scale;
        
        if (is_causal && jj > i) {
            score_extra = -static_cast<T>(1e30);
        }
        
        int score_idx_extra = batch * query_heads * target_seq_len * src_seq_len +
                             q_head * target_seq_len * src_seq_len +
                             i * src_seq_len + jj;
        scores[score_idx_extra] = score_extra;
    }
}

template<typename T>
__global__ void applySoftmax(
    const T* scores, T* weights,
    int batch_size, int query_heads, int target_seq_len, int src_seq_len,
    bool is_causal) {
    
    int batch = blockIdx.x;
    int q_head = blockIdx.y;
    int i = blockIdx.z;
    
    if (batch >= batch_size || q_head >= query_heads || i >= target_seq_len) return;
    
    int base_idx = batch * query_heads * target_seq_len * src_seq_len +
                   q_head * target_seq_len * src_seq_len +
                   i * src_seq_len;
    
    // Find max for numerical stability
    T max_score = -static_cast<T>(1e30);
    int valid_count = 0;
    for (int j = 0; j < src_seq_len; ++j) {
        if (!is_causal || j <= i) {
            T current_score = scores[base_idx + j];
            max_score = (max_score > current_score) ? max_score : current_score;
            valid_count++;
        }
    }
    
    // Handle case where all positions are masked
    if (valid_count == 0) {
        for (int j = 0; j < src_seq_len; ++j) {
            weights[base_idx + j] = static_cast<T>(0);
        }
        return;
    }
    
    // Compute exp and sum
    T sum_exp = static_cast<T>(0);
    for (int j = 0; j < src_seq_len; ++j) {
        if (is_causal && j > i) {
            weights[base_idx + j] = static_cast<T>(0);
        } else {
            T exp_val;
            if constexpr (std::is_same_v<T, float>) {
                exp_val = expf(scores[base_idx + j] - max_score);
            } else {
                exp_val = exp(scores[base_idx + j] - max_score);
            }
            weights[base_idx + j] = exp_val;
            sum_exp += exp_val;
        }
    }
    
    // Normalize
    if (sum_exp > static_cast<T>(0)) {
        T inv_sum = static_cast<T>(1) / sum_exp;
        for (int j = 0; j < src_seq_len; ++j) {
            weights[base_idx + j] *= inv_sum;
        }
    }
}

template<typename T>
__global__ void computeOutput(
    const T* weights, const T* v, T* output,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim) {
    
    int batch = blockIdx.x;
    int q_head = blockIdx.y;
    int i = blockIdx.z;
    int d = threadIdx.x;
    
    if (batch >= batch_size || q_head >= query_heads || 
        i >= target_seq_len || d >= head_dim) return;
    
    int head_group_size = query_heads / kv_heads;
    int kv_head = q_head / head_group_size;
    
    T output_val = static_cast<T>(0);
    
    int weight_base = batch * query_heads * target_seq_len * src_seq_len +
                     q_head * target_seq_len * src_seq_len +
                     i * src_seq_len;
    
    for (int j = 0; j < src_seq_len; ++j) {
        int v_idx = batch * src_seq_len * kv_heads * head_dim + 
                   j * kv_heads * head_dim + kv_head * head_dim + d;
        
        output_val += weights[weight_base + j] * v[v_idx];
    }
    
    int o_idx = batch * target_seq_len * query_heads * head_dim + 
               i * query_heads * head_dim + q_head * head_dim + d;
    output[o_idx] = output_val;
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    T scale = static_cast<T>(1.0) / static_cast<T>(sqrt(static_cast<double>(head_dim)));
    
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t v_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t scores_size = batch_size * query_heads * target_seq_len * src_seq_len;
    
    // Allocate GPU memory
    T *d_q, *d_k, *d_v, *d_o, *d_scores, *d_weights;
    
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, k_size * sizeof(T));
    cudaMalloc(&d_v, v_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));
    cudaMalloc(&d_scores, scores_size * sizeof(T));
    cudaMalloc(&d_weights, scores_size * sizeof(T));
    
    // Copy input data to GPU
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice);
    
    // Launch kernels
    dim3 grid1(batch_size, query_heads, target_seq_len);
    dim3 block1(min(src_seq_len, 1024));
    computeAttentionScores<<<grid1, block1>>>(
        d_q, d_k, d_scores, batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, scale, is_causal);
    
    dim3 grid2(batch_size, query_heads, target_seq_len);
    applySoftmax<<<grid2, 1>>>(
        d_scores, d_weights, batch_size, query_heads, 
        target_seq_len, src_seq_len, is_causal);
    
    dim3 grid3(batch_size, query_heads, target_seq_len);
    dim3 block3(min(head_dim, 1024));
    computeOutput<<<grid3, block3>>>(
        d_weights, d_v, d_o, batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim);
    
    // Copy result back to host
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_scores);
    cudaFree(d_weights);
    
    cudaDeviceSynchronize();
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
