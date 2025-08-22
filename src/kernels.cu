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
    
    // 第一阶段：原子计数 - 统计小于、等于和大于pivot的元素数量
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

    // 第二阶段：实际分区 - 将元素分类到临时数组中
    for (size_t i = tid; i < n; i += stride) {
        if (data[i] < pivot) {
            int pos = atomicAdd(less_idx, 1);
            temp[pos] = data[i];
        } else if (data[i] > pivot) {
            int pos = atomicAdd(greater_idx, 1);
            temp[n - 1 - pos] = data[i]; // 反向存储以便后续合并
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

    // 使用快速选择算法查找第k大元素
    while (left < right) {
        // 1. 选择pivot（简单选择中间元素，可改进为随机选择）
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

        // 6. 决定下一步搜索范围
        if (target_k <= greater) {
            // 第k大元素在较大分区
            left = left + less + equal;
        } else if (target_k <= greater + equal) {
            // 找到目标元素
            result = pivot;
            break;
        } else {
            // 第k大元素在较小分区
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
struct OnlineState {
    T max_val;
    T sum_exp;
    
    __device__ OnlineState() : max_val(-1e30f), sum_exp(0.0f) {}
};

template<typename T>
__device__ void updateState(OnlineState<T>& state, T new_max, T new_sum) {
    // 在线更新最大值和指数和，保持数值稳定性
    if (new_max > state.max_val) {
        T scale = expf(state.max_val - new_max);
        state.sum_exp = fmaf(state.sum_exp, scale, new_sum);
        state.max_val = new_max;
    } else {
        T scale = expf(new_max - state.max_val);
        state.sum_exp = fmaf(new_sum, scale, state.sum_exp);
    }
}

// Warp级别的归约操作
template<typename T>
__device__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block级别的归约操作
template<typename T>
__device__ T blockReduceMax(T val, T* shared_mem) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp内归约
    val = warpReduceMax(val);
    
    // 每个warp的结果存到shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    // 最后一个warp处理所有warp的结果
    if (warp_id == 0) {
        val = (lane_id < (blockDim.x + 31) / 32) ? shared_mem[lane_id] : -1e30f;
        val = warpReduceMax(val);
    }
    
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val, T* shared_mem) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp内归约
    val = warpReduceSum(val);
    
    // 每个warp的结果存到shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    // 最后一个warp处理所有warp的结果
    if (warp_id == 0) {
        val = (lane_id < (blockDim.x + 31) / 32) ? shared_mem[lane_id] : 0.0f;
        val = warpReduceSum(val);
    }
    
    return val;
}

// Flash Attention核心算法实现
template<typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ output,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    T scale, bool is_causal, int kv_block_size) {
    
    int batch = blockIdx.x;
    int q_head = blockIdx.y;
    int i = blockIdx.z;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || q_head >= query_heads || i >= target_seq_len) return;
    
    // 计算对应的KV头（支持分组查询注意力）
    int head_group_size = query_heads / kv_heads;
    int kv_head = q_head / head_group_size;
    
    // 计算有效的K/V长度（因果掩码时限制为当前位置）
    int valid_kv_len = is_causal ? (i + 1) : src_seq_len;
    int num_blocks = (valid_kv_len + kv_block_size - 1) / kv_block_size;
    
    // Shared memory布局
    extern __shared__ T shared_mem[];
    T* shared_reduction = shared_mem;
    T* shared_scores = shared_reduction + ((blockDim.x + 31) / 32);
    T* shared_output = shared_scores + kv_block_size;
    
    // 初始化输出累加器
    for (int d = tid; d < head_dim; d += blockDim.x) {
        shared_output[d] = 0.0f;
    }
    __syncthreads();
    
    // Flash Attention状态（在线softmax）
    OnlineState<T> flash_state;
    
    // Q的基础索引
    int q_base = batch * target_seq_len * query_heads * head_dim + 
                 i * query_heads * head_dim + q_head * head_dim;
    
    // 分块处理K/V序列
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int kv_start = block_idx * kv_block_size;
        int kv_end = min(kv_start + kv_block_size, valid_kv_len);
        int current_block_size = kv_end - kv_start;
        
        if (current_block_size <= 0) break;
        
        // 步骤1：计算当前块的attention scores
        for (int j = tid; j < current_block_size; j += blockDim.x) {
            int global_j = kv_start + j;
            T score = 0.0f;
            
            // 使用Kahan求和提高精度
            T c = 0.0f; // 补偿项
            for (int d = 0; d < head_dim; ++d) {
                T q_val = q[q_base + d];
                T k_val = k[batch * src_seq_len * kv_heads * head_dim + 
                           global_j * kv_heads * head_dim + kv_head * head_dim + d];
                
                T product = q_val * k_val;
                T y = product - c;
                T t = score + y;
                c = (t - score) - y;
                score = t;
            }
            
            score *= scale;
            
            // 应用因果掩码
            if (is_causal && global_j > i) {
                score = -1e30f;
            }
            
            shared_scores[j] = score;
        }
        __syncthreads();
        
        // 步骤2：找到块内最大值（用于数值稳定的softmax）
        T block_max = -1e30f;
        for (int j = tid; j < current_block_size; j += blockDim.x) {
            block_max = fmaxf(block_max, shared_scores[j]);
        }
        block_max = blockReduceMax(block_max, shared_reduction);
        
        // 步骤3：计算exp和
        T block_sum = 0.0f;
        for (int j = tid; j < current_block_size; j += blockDim.x) {
            T exp_score = expf(shared_scores[j] - block_max);
            shared_scores[j] = exp_score;
            block_sum += exp_score;
        }
        block_sum = blockReduceSum(block_sum, shared_reduction);
        __syncthreads();
        
        // 步骤4：更新全局状态
        OnlineState<T> old_state = flash_state;
        if (tid == 0) {
            updateState(flash_state, block_max, block_sum);
        }
        
        // 广播状态到所有线程
        if (tid == 0) {
            shared_reduction[0] = flash_state.max_val;
            shared_reduction[1] = flash_state.sum_exp;
            shared_reduction[2] = old_state.max_val;
            shared_reduction[3] = old_state.sum_exp;
        }
        __syncthreads();
        
        T new_max = shared_reduction[0];
        T new_sum = shared_reduction[1];
        T old_max = shared_reduction[2];
        T old_sum = shared_reduction[3];
        
        flash_state.max_val = new_max;
        flash_state.sum_exp = new_sum;
        
        // 步骤5：重新缩放之前的输出
        T rescale_factor = (old_sum > 1e-10f) ? expf(old_max - new_max) : 0.0f;
        
        for (int d = tid; d < head_dim; d += blockDim.x) {
            shared_output[d] *= rescale_factor;
        }
        __syncthreads();
        
        // 步骤6：累加当前块的贡献
        for (int d = tid; d < head_dim; d += blockDim.x) {
            T weighted_sum = 0.0f;
            T c = 0.0f; // Kahan求和补偿项
            
            for (int j = 0; j < current_block_size; ++j) {
                int global_j = kv_start + j;
                T weight = shared_scores[j];
                
                T v_val = v[batch * src_seq_len * kv_heads * head_dim + 
                           global_j * kv_heads * head_dim + kv_head * head_dim + d];
                
                T product = weight * v_val;
                T y = product - c;
                T t = weighted_sum + y;
                c = (t - weighted_sum) - y;
                weighted_sum = t;
            }
            
            shared_output[d] += weighted_sum;
        }
        __syncthreads();
    }
    
    // 步骤7：最终归一化并写入输出
    T inv_sum = (flash_state.sum_exp > 1e-10f) ? (1.0f / flash_state.sum_exp) : 0.0f;
    
    int output_base = batch * target_seq_len * query_heads * head_dim + 
                     i * query_heads * head_dim + q_head * head_dim;
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        output[output_base + d] = shared_output[d] * inv_sum;
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    T scale = static_cast<T>(1.0) / sqrtf(static_cast<T>(head_dim));
    
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t v_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
    
    // GPU内存分配
    T *d_q, *d_k, *d_v, *d_o;
    
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, k_size * sizeof(T));
    cudaMalloc(&d_v, v_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));
    
    // 初始化输出为0
    cudaMemset(d_o, 0, o_size * sizeof(T));
    
    // 拷贝输入数据
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice);
    
    // 动态调整KV块大小
    int kv_block_size = min(128, src_seq_len);
    if (src_seq_len <= 512) {
        kv_block_size = min(64, src_seq_len);
    }
    
    // 计算shared memory需求
    size_t warp_count = (128 + 31) / 32;
    size_t shared_mem_size = (warp_count + kv_block_size + head_dim) * sizeof(T);
    
    // 启动kernel
    dim3 grid(batch_size, query_heads, target_seq_len);
    dim3 block(128);
    
    flashAttentionKernel<<<grid, block, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        scale, is_causal, kv_block_size
    );
    
    // 拷贝结果
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    
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