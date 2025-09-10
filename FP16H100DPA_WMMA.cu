#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <map>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// 内核函数：使用Tensor Core计算点积加
__global__ void dpas_kernel(const half* A, const half* B, const float* C, float* D, int num_lines, int* round_mode) {
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

    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half smem_A[16*16];
    __shared__ half smem_B[16*16];
    __shared__ float result_smem[16*16];

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

    // 执行矩阵乘加
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // 存储结果到共享内存
    wmma::store_matrix_sync(result_smem, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    float dot_product = result_smem[0];  // 获取点积结果

    // 根据舍入模式计算最终结果
    float final_result;
    switch (my_round_mode) {
        case 0:  // RND_NEAREST
            final_result = __fadd_rn(dot_product, myC);
            break;
        case 1:  // RND_ZERO
            final_result = __fadd_rz(dot_product, myC);
            break;
        default:
            final_result = __fadd_rn(dot_product, myC);
    }

    if (threadIdx.x == 0) {
        D[line_id] = final_result;
    }
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
        {"RND_ZERO", 1},
        {"RND_NEAREST", 0}
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
        std::cerr << "Failed to open input file." << std::endl;
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
            // 去除 token 前后的空格
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
            std::cerr << "Unknown rounding mode: " << tokens[1] << ", using RND_NEAREST as default." << std::endl;
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
    float *d_C, *d_D;
    int *d_round_mode;

    size_t A_size = num_lines * 8 * sizeof(half);
    size_t B_size = num_lines * 8 * sizeof(half);
    size_t C_size = num_lines * sizeof(float);
    size_t D_size = num_lines * sizeof(float);
    size_t round_mode_size = num_lines * sizeof(int);

    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_B, B_size);
    cudaMalloc(&d_C, C_size);
    cudaMalloc(&d_D, D_size);
    cudaMalloc(&d_round_mode, round_mode_size);

    // 拷贝数据到设备
    cudaMemcpy(d_A, A_vec.data(), A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_vec.data(), B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_vec.data(), C_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_round_mode, round_modes.data(), round_mode_size, cudaMemcpyHostToDevice);

    // 启动内核
    dim3 blocks(num_lines);
    dim3 threads(32);  // 一个warp

    dpas_kernel<<<blocks, threads>>>(d_A, d_B, d_C, d_D, num_lines, d_round_mode);

    cudaDeviceSynchronize();

    // 检查内核执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // 拷贝结果回主机
    std::vector<float> D_vec(num_lines);
    cudaMemcpy(D_vec.data(), d_D, D_size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_round_mode);

    // 重新读取输入文件，并将结果追加到每行末尾
    std::ifstream infile2(input_filename);
    std::ofstream outfile(output_filename);
    if (!infile2.is_open() || !outfile.is_open()) {
        std::cerr << "Failed to open files for writing." << std::endl;
        return 1;
    }

    int index = 0;
    while (std::getline(infile2, line)) {
        // 移除行末的换行符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        // 追加结果（十六进制格式）
        outfile << line << ", " << float2hex(D_vec[index]) << std::endl;
        index++;
    }

    infile2.close();
    outfile.close();

    std::cout << "Processing completed. Output written to " << output_filename << std::endl;
    return 0;
}
