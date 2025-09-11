#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <map>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

// 检查CUDA错误
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d : %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// 检查cuBLAS错误
#define checkCublasErrors(err) __checkCublasErrors (err, __FILE__, __LINE__)
inline void __checkCublasErrors(cublasStatus_t err, const char *file, const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "cuBLAS Error at %s:%d : %d\n", file, line, err);
        exit(EXIT_FAILURE);
    }
}

// 内核函数1：使用WMMA计算点积加
__global__ void dpas_kernel_wmma(const half* A, const half* B, const float* C, float* D, int num_lines, int* round_mode) {
    int line_id = blockIdx.x;
    if (line_id >= num_lines) return;

    const half* myA = A + line_id * 8;
    const half* myB = B + line_id * 8;
    float myC = C[line_id];
    int my_round_mode = round_mode[line_id];

    // 声明WMMA片段
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // 初始化累加器为C的值
    wmma::fill_fragment(acc_frag, 0.0f);
    if (threadIdx.x == 0) {
        acc_frag.x[0] = myC;
    }

    __shared__ half smem_A[16*16];
    __shared__ half smem_B[16*16];

    // 初始化共享内存为0
    for (int i = threadIdx.x; i < 16*16; i += blockDim.x) {
        smem_A[i] = __float2half(0.0f);
        smem_B[i] = __float2half(0.0f);
    }
    __syncthreads();

    // 线程0加载A和B数据到共享内存
    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            smem_A[i] = myA[i];  // 行优先，第一行前8个元素
            smem_B[i] = myB[i];  // 列优先，第一列的前8个元素
        }
    }
    __syncthreads();

    // 加载矩阵到片段
    wmma::load_matrix_sync(a_frag, smem_A, 16);
    wmma::load_matrix_sync(b_frag, smem_B, 16);

    // 执行矩阵乘加 (A*B + C)
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // 存储结果到共享内存
    __shared__ float result_smem[16*16];
    wmma::store_matrix_sync(result_smem, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    float dot_product = result_smem[0];  // 获取点积结果

    if (threadIdx.x == 0) {
        D[line_id] = dot_product;
    }
}

// 内核函数2：使用CUDA Core计算点积加（精确实现）
__global__ void dpas_kernel_cudacore(const half* A, const half* B, const float* C, float* D, int num_lines, int* round_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_lines) return;

    const half* myA = A + idx * 8;
    const half* myB = B + idx * 8;
    float myC = C[idx];
    int my_round_mode = round_mode[idx];
    
    // 使用Kahan求和算法提高精度
    float sum = 0.0f;
    float c = 0.0f; // 补偿值
    
    for (int i = 0; i < 8; i++) {
        float a_val = __half2float(myA[i]);
        float b_val = __half2float(myB[i]);
        float product = a_val * b_val;
        
        // Kahan求和
        float y = product - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    D[idx] = sum + myC;
}

// 使用cuBLAS执行八点积加操作
void executeDot8WithCublas(const half* d_A, const half* d_B, const float* d_C, float* d_D, int numTests) {
    cublasHandle_t handle;
    checkCublasErrors(cublasCreate(&handle));
    
    // 设置矩阵乘法参数
    float alpha = 1.0f;
    float beta = 1.0f;
    
    // 执行矩阵乘法 (1x8) * (8x1) + (1x1) = (1x1)
    for (int i = 0; i < numTests; i++) {
        // 使用cublasGemmEx执行混合精度矩阵乘法
        checkCublasErrors(cublasGemmEx(
            handle,
            CUBLAS_OP_N,  // A矩阵不需要转置
            CUBLAS_OP_N,  // B矩阵不需要转置
            1,            // 结果矩阵的行数
            1,            // 结果矩阵的列数
            8,            // 公共维度
            &alpha,       // alpha值
            d_A + i * 8,  // A矩阵数据
            CUDA_R_16F,   // A矩阵数据类型
            1,            // A矩阵的leading dimension
            d_B + i * 8,  // B矩阵数据
            CUDA_R_16F,   // B矩阵数据类型
            8,            // B矩阵的leading dimension
            &beta,        // beta值
            d_D + i,      // 输出矩阵数据
            CUDA_R_32F,   // 输出矩阵数据类型
            1,            // 输出矩阵的leading dimension
            CUBLAS_COMPUTE_32F,  // 计算类型
            CUBLAS_GEMM_DEFAULT_TENSOR_OP  // 算法
        ));
    }
    
    // 同步等待所有操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCublasErrors(cublasDestroy(handle));
}

// 将十六进制字符串转换为half
half hex2half(const std::string& s) {
    unsigned short val;
    std::stringstream ss;
    ss << std::hex << s;
    ss >> val;
    return __short_as_half(val);
}

// 将十六进制字符串转换为float
float hex2float(const std::string& s) {
    unsigned int val;
    std::stringstream ss;
    ss << std::hex << s;
    ss >> val;
    float result;
    std::memcpy(&result, &val, sizeof(val));
    return result;
}

// 将float转换为十六进制字符串
std::string float2hex(float f) {
    unsigned int u;
    std::memcpy(&u, &f, sizeof(f));
    std::stringstream ss;
    ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << u;
    return ss.str();
}

int main() {
    std::map<std::string, int> roundModeMap = {
        {"RND_NEAREST", 0},
        {"RND_ZERO", 1},
        {"RND_FINITE", 2},
        {"RND_INF", 3}
    };
    
    std::string input_filename, output_filename;
    
    // 获取输入文件名
    while (true) {
        std::cout << "请输入输入文件名 (默认: fp16_pda_input.txt): ";
        std::getline(std::cin, input_filename);
        
        if (input_filename.empty()) {
            input_filename = "fp16_pda_input.txt";
        }
        
        std::ifstream testFile(input_filename);
        if (testFile.good()) {
            testFile.close();
            break;
        }
        
        std::cout << "文件 " << input_filename << " 不存在，请重新输入。\n";
    }
    
    // 获取输出文件名
    std::cout << "请输入输出文件名 (默认: fp16_pda_output.txt): ";
    std::getline(std::cin, output_filename);
    if (output_filename.empty()) {
        output_filename = "fp16_pda_output.txt";
    }

    std::ifstream infile(input_filename);
    if (!infile.is_open()) {
        std::cerr << "打开输入文件失败" << std::endl;
        return 1;
    }

    std::vector<int> round_modes;
    std::vector<half> A_vec;
    std::vector<half> B_vec;
    std::vector<float> C_vec;
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, ',')) {
            token.erase(0, token.find_first_not_of(" "));
            token.erase(token.find_last_not_of(" ") + 1);
            tokens.push_back(token);
        }
        if (tokens.size() < 19) {
            std::cerr << "Invalid line: " << line << std::endl;
            continue;
        }

        // 解析舍入模式
        int round_mode;
        if (roundModeMap.find(tokens[1]) != roundModeMap.end()) {
            round_mode = roundModeMap[tokens[1]];
        } else {
            std::cerr << "未知舍入模式：" << tokens[1] << "，使用默认舍入模式：RND_NEAREST " << std::endl;
            round_mode = 0;
        }
        round_modes.push_back(round_mode);

        // 解析A的8个元素
        for (int i = 0; i < 8; i++) {
            std::string hex_str = tokens[2 + i];
            half h = hex2half(hex_str);
            A_vec.push_back(h);
        }

        // 解析B的8个元素
        for (int i = 0; i < 8; i++) {
            std::string hex_str = tokens[10 + i];
            half h = hex2half(hex_str);
            B_vec.push_back(h);
        }

        // 解析C
        std::string hex_c = tokens[18];
        float c_val = hex2float(hex_c);
        C_vec.push_back(c_val);
    }

    infile.close();
    int num_lines = round_modes.size();

    if (num_lines == 0) {
        std::cerr << "错误：无有效测试用例，程序终止。\n";
        return 1;
    }

    std::cout << "找到 " << num_lines << " 个测试用例，开始处理...\n";

    // 分配设备内存
    half *d_A, *d_B;
    float *d_C, *d_D_wmma, *d_D_cudacore, *d_D_cublas;
    int *d_round_mode;

    size_t A_size = num_lines * 8 * sizeof(half);
    size_t B_size = num_lines * 8 * sizeof(half);
    size_t C_size = num_lines * sizeof(float);
    size_t D_size = num_lines * sizeof(float);
    size_t round_mode_size = num_lines * sizeof(int);

    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_B, B_size);
    cudaMalloc(&d_C, C_size);
    cudaMalloc(&d_D_wmma, D_size);
    cudaMalloc(&d_D_cudacore, D_size);
    cudaMalloc(&d_D_cublas, D_size);
    cudaMalloc(&d_round_mode, round_mode_size);

    // 拷贝数据到设备
    cudaMemcpy(d_A, A_vec.data(), A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_vec.data(), B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_vec.data(), C_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_round_mode, round_modes.data(), round_mode_size, cudaMemcpyHostToDevice);

    // 启动WMMA内核
    dim3 blocks_wmma(num_lines);
    dim3 threads_wmma(32);  // 一个warp

    dpas_kernel_wmma<<<blocks_wmma, threads_wmma>>>(d_A, d_B, d_C, d_D_wmma, num_lines, d_round_mode);

    // 启动CUDA Core内核
    int block_size = 256;
    int grid_size = (num_lines + block_size - 1) / block_size;
    
    dpas_kernel_cudacore<<<grid_size, block_size>>>(d_A, d_B, d_C, d_D_cudacore, num_lines, d_round_mode);

    // 使用cuBLAS计算
    executeDot8WithCublas(d_A, d_B, d_C, d_D_cublas, num_lines);

    cudaDeviceSynchronize();

    // 检查内核执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // 拷贝结果回主机
    std::vector<float> D_vec_wmma(num_lines);
    std::vector<float> D_vec_cudacore(num_lines);
    std::vector<float> D_vec_cublas(num_lines);
    cudaMemcpy(D_vec_wmma.data(), d_D_wmma, D_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(D_vec_cudacore.data(), d_D_cudacore, D_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(D_vec_cublas.data(), d_D_cublas, D_size, cudaMemcpyDeviceToHost);

    // 比较WMMA和cuBLAS分别与CUDA Core的精度
    int differences_wmma_cc = 0;
    int differences_cublas_cc = 0;
    float max_diff_wmma_cc = 0.0f;
    float max_diff_cublas_cc = 0.0f;
    
    for (int i = 0; i < num_lines; i++) {
        float diff_wmma_cc = fabs(D_vec_wmma[i] - D_vec_cudacore[i]);
        float diff_cublas_cc = fabs(D_vec_cublas[i] - D_vec_cudacore[i]);
        
        if (diff_wmma_cc > 1e-6) {
            differences_wmma_cc++;
            if (diff_wmma_cc > max_diff_wmma_cc) max_diff_wmma_cc = diff_wmma_cc;
        }
        
        if (diff_cublas_cc > 1e-6) {
            differences_cublas_cc++;
            if (diff_cublas_cc > max_diff_cublas_cc) max_diff_cublas_cc = diff_cublas_cc;
        }
    }
    
    std::cout << "精度比较结果:" << std::endl;
    std::cout << "WMMA vs CUDA Core: " << differences_wmma_cc << " 个测试用例有显著差异，最大差异: " << max_diff_wmma_cc << std::endl;
    std::cout << "cuBLAS vs CUDA Core: " << differences_cublas_cc << " 个测试用例有显著差异，最大差异: " << max_diff_cublas_cc << std::endl;

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D_wmma);
    cudaFree(d_D_cudacore);
    cudaFree(d_D_cublas);
    cudaFree(d_round_mode);

    // 重新读取输入文件，并将结果追加到每行末尾
    std::ifstream infile2(input_filename);
    std::ofstream outfile(output_filename);
    if (!infile2.is_open() || !outfile.is_open()) {
        std::cerr << "写入输出文件失败" << std::endl;
        return 1;
    }

    int index = 0;
    while (std::getline(infile2, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        // 追加三种方法的结果（十六进制格式）
        outfile << line << ", " << float2hex(D_vec_wmma[index]) 
                << ", " << float2hex(D_vec_cudacore[index])
                << ", " << float2hex(D_vec_cublas[index]) << std::endl;
        index++;
    }

    infile2.close();
    outfile.close();

    std::cout << "八点积加操作完成，输出文件：" << output_filename << std::endl;
    std::cout << "输出文件中包含三种方法的结果，倒数第三列是WMMA结果，倒数第二列是CUDA Core结果，最后一列是cuBLAS结果" << std::endl;
    
    return 0;
}
