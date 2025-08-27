#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>

// 检查CUDA错误
#define CHECK_CUDA(status) \
    do { \
        cudaError_t err = status; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while (0)

// 检查cuBLAS错误
#define CHECK_CUBLAS(status) \
    do { \
        cublasStatus_t err = status; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": " << err << std::endl; \
            return false; \
        } \
    } while (0)

// 将十六进制字符串转换为unsigned short（用于FP16）
bool hexStringToFP16(const std::string& str, unsigned short& value) {
    try {
        value = static_cast<unsigned short>(std::stoul(str, nullptr, 16));
        return true;
    } catch (...) {
        return false;
    }
}

// 将十六进制字符串转换为unsigned int（用于FP32）
bool hexStringToFP32(const std::string& str, unsigned int& value) {
    try {
        value = std::stoul(str, nullptr, 16);
        return true;
    } catch (...) {
        return false;
    }
}

// 将float转换为十六进制字符串
std::string floatToHexString(float f) {
    unsigned int u;
    std::memcpy(&u, &f, sizeof(float));
    std::stringstream ss;
    ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << u;
    return ss.str();
}

// 修剪字符串中的空格
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

// 处理一行数据：计算点积加
bool processLine(const std::string& line, cublasHandle_t handle, std::string& outputLine) {
    // 分割字符串 by comma
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(trim(token));
    }

    if (tokens.size() != 19) {
        std::cerr << "Invalid number of tokens in line: " << tokens.size() << std::endl;
        return false;
    }

    // 解析A向量（8个FP16值）
    std::vector<__half> A_h(8);
    for (int i = 0; i < 8; i++) {
        unsigned short value;
        if (!hexStringToFP16(tokens[2+i], value)) {
            std::cerr << "Failed to parse A[" << i << "]: " << tokens[2+i] << std::endl;
            return false;
        }
        A_h[i] = __half_raw{value};
    }

    // 解析B向量（8个FP16值）
    std::vector<__half> B_h(8);
    for (int i = 0; i < 8; i++) {
        unsigned short value;
        if (!hexStringToFP16(tokens[10+i], value)) {
            std::cerr << "Failed to parse B[" << i << "]: " << tokens[10+i] << std::endl;
            return false;
        }
        B_h[i] = __half_raw{value};
    }

    // 解析C标量（FP32值）
    unsigned int C_uint;
    if (!hexStringToFP32(tokens[18], C_uint)) {
        std::cerr << "Failed to parse C: " << tokens[18] << std::endl;
        return false;
    }
    float C_h;
    std::memcpy(&C_h, &C_uint, sizeof(float));

    // 分配设备内存
    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float* d_result = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, 8 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, 8 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, A_h.data(), 8 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_h.data(), 8 * sizeof(__half), cudaMemcpyHostToDevice));

    // 使用cuBLAS计算点积
    float dot_result = 0.0f;
    
    // 将FP16转换为FP32进行计算
    std::vector<float> A_f32(8), B_f32(8);
    for (int i = 0; i < 8; i++) {
        A_f32[i] = __half2float(A_h[i]);
        B_f32[i] = __half2float(B_h[i]);
    }
    
    // 分配设备内存用于FP32计算
    float* d_A_f32 = nullptr;
    float* d_B_f32 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A_f32, 8 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_f32, 8 * sizeof(float)));
    
    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_A_f32, A_f32.data(), 8 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_f32, B_f32.data(), 8 * sizeof(float), cudaMemcpyHostToDevice));
    
    // 使用cuBLAS计算点积
    CHECK_CUBLAS(cublasSdot(handle, 8, d_A_f32, 1, d_B_f32, 1, &dot_result));
    
    // 加上C值
    float result = dot_result + C_h;
    
    // 清理设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_A_f32);
    cudaFree(d_B_f32);
    cudaFree(d_result);

    // 构建输出行
    outputLine = line + ", " + floatToHexString(result);

    return true;
}

int main() {
    // 初始化cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 获取输入和输出文件名
    std::string inputFileName, outputFileName;
    std::cout << "Enter input file name: ";
    std::cin >> inputFileName;
    std::cout << "Enter output file name: ";
    std::cin >> outputFileName;

    // 打开输入文件
    std::ifstream inputFile(inputFileName);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file: " << inputFileName << std::endl;
        return 1;
    }

    // 打开输出文件
    std::ofstream outputFile(outputFileName);
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFileName << std::endl;
        return 1;
    }

    // 逐行处理
    std::string line;
    while (std::getline(inputFile, line)) {
        std::string outputLine;
        if (processLine(line, handle, outputLine)) {
            outputFile << outputLine << std::endl;
        } else {
            std::cerr << "Failed to process line: " << line << std::endl;
        }
    }

    // 清理
    inputFile.close();
    outputFile.close();
    cublasDestroy(handle);

    std::cout << "Processing completed. Output written to " << outputFileName << std::endl;
    return 0;
}
