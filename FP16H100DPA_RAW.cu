#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_fp16.h>

// 将十六进制字符串转换为半精度浮点数
__half hexStringToHalf(const std::string& hexStr) {
    uint16_t hexValue = std::stoul(hexStr, nullptr, 16);
    return *reinterpret_cast<__half*>(&hexValue);
}

// 将十六进制字符串转换为单精度浮点数
float hexStringToFloat(const std::string& hexStr) {
    uint32_t hexValue = std::stoul(hexStr, nullptr, 16);
    return *reinterpret_cast<float*>(&hexValue);
}

// 将单精度浮点数转换为十六进制字符串
std::string floatToHexString(float f) {
    uint32_t hexValue = *reinterpret_cast<uint32_t*>(&f);
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << hexValue;
    return ss.str();
}

// CUDA内核：计算8点积加
__global__ void dot8Kernel(const __half* a, const __half* b, float c, float* result) {
    // 使用共享内存存储中间结果
    __shared__ float shared_sum;
    
    if (threadIdx.x == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    
    // 每个线程处理一个元素对
    int idx = threadIdx.x;
    if (idx < 8) {
        float a_val = __half2float(a[idx]);
        float b_val = __half2float(b[idx]);
        atomicAdd(&shared_sum, a_val * b_val);
    }
    __syncthreads();
    
    // 第一个线程将结果加上c
    if (threadIdx.x == 0) {
        *result = shared_sum + c;
    }
}

// 执行八点积加计算
cudaError_t computeDot8(const std::vector<__half>& a, const std::vector<__half>& b, float c, float& result) {
    // 准备设备内存
    __half *d_a, *d_b;
    float *d_c, *d_result;
    
    cudaMalloc(&d_a, 8 * sizeof(__half));
    cudaMalloc(&d_b, 8 * sizeof(__half));
    cudaMalloc(&d_c, sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_a, a.data(), 8 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), 8 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动内核
    dot8Kernel<<<1, 8>>>(d_a, d_b, c, d_result);
    
    // 等待内核完成
    cudaError_t kernelStatus = cudaDeviceSynchronize();
    if (kernelStatus != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_result);
        return kernelStatus;
    }
    
    // 拷贝结果回主机
    cudaError_t copyStatus = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_result);
    
    return copyStatus;
}

// 解析输入行
bool parseLine(const std::string& line, std::string& op, std::string& rounding,
               std::vector<__half>& a, std::vector<__half>& b, float& c) {
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    
    while (std::getline(iss, token, ',')) {
        // 去除前后空格
        size_t start = token.find_first_not_of(" ");
        size_t end = token.find_last_not_of(" ");
        if (start != std::string::npos && end != std::string::npos) {
            tokens.push_back(token.substr(start, end - start + 1));
        } else {
            tokens.push_back("");
        }
    }
    
    if (tokens.size() < 19) return false;
    
    op = tokens[0];
    rounding = tokens[1];
    
    a.resize(8);
    for (int i = 0; i < 8; i++) {
        a[i] = hexStringToHalf(tokens[2 + i]);
    }
    
    b.resize(8);
    for (int i = 0; i < 8; i++) {
        b[i] = hexStringToHalf(tokens[10 + i]);
    }
    
    c = hexStringToFloat(tokens[18]);
    
    return true;
}

int main() {
    std::string inputFile, outputFile;
    
    // 询问输入文件名
    std::cout << "请输入输入文件名: ";
    std::cin >> inputFile;
    
    // 询问输出文件名
    std::cout << "请输入输出文件名: ";
    std::cin >> outputFile;
    
    // 打开输入文件
    std::ifstream fin(inputFile);
    if (!fin.is_open()) {
        std::cerr << "无法打开输入文件: " << inputFile << std::endl;
        return 1;
    }
    
    // 打开输出文件
    std::ofstream fout(outputFile);
    if (!fout.is_open()) {
        std::cerr << "无法打开输出文件: " << outputFile << std::endl;
        fin.close();
        return 1;
    }
    
    // 处理每一行
    std::string line;
    int lineNum = 0;
    while (std::getline(fin, line)) {
        lineNum++;
        
        std::string op, rounding;
        std::vector<__half> a, b;
        float c;
        
        if (!parseLine(line, op, rounding, a, b, c)) {
            std::cerr << "解析第 " << lineNum << " 行时出错" << std::endl;
            continue;
        }
        
        if (op != "MPDPA") {
            std::cerr << "第 " << lineNum << " 行不支持的操作: " << op << std::endl;
            continue;
        }
        
        if (rounding != "RND_NEAREST") {
            std::cerr << "第 " << lineNum << " 行不支持的舍入模式: " << rounding << std::endl;
            continue;
        }
        
        float result;
        cudaError_t status = computeDot8(a, b, c, result);
        
        if (status != cudaSuccess) {
            std::cerr << "第 " << lineNum << " 行计算失败: " 
                      << cudaGetErrorString(status) << std::endl;
            continue;
        }
        
        // 写入结果到输出文件
        fout << line << ", 0x" << floatToHexString(result) << std::endl;
    }
    
    fin.close();
    fout.close();
    
    std::cout << "处理完成。结果已写入 " << outputFile << std::endl;
    
    return 0;
}
