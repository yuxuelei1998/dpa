#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_bf16.h>

// 将十六进制字符串转换为BF16浮点数
__nv_bfloat16 hexStringToBF16(const std::string& hexStr) {
    uint16_t hexValue = std::stoul(hexStr, nullptr, 16);
    return *reinterpret_cast<__nv_bfloat16*>(&hexValue);
}

// 将十六进制字符串转换为单精度浮点数
float hexStringToFloat(const std::string& hexStr) {
    uint32_t hexValue = std::stoul(hexStr, nullptr, 16);
    return *reinterpret_cast<float*>(&hexValue);
}

// 将BF16浮点数转换为十六进制字符串
std::string bf16ToHexString(__nv_bfloat16 bf) {
    uint16_t hexValue = *reinterpret_cast<uint16_t*>(&bf);
    std::stringstream ss;
    ss << std::hex << std::setw(4) << std::setfill('0') << hexValue;
    return ss.str();
}

// 将单精度浮点数转换为十六进制字符串
std::string floatToHexString(float f) {
    uint32_t hexValue = *reinterpret_cast<uint32_t*>(&f);
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << hexValue;
    return ss.str();
}

// CUDA内核：计算8点积加（BF16版本）
__global__ void dot8BF16Kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, float c, float* result) {
    // 使用共享内存存储中间结果
    __shared__ float shared_sum;
    
    if (threadIdx.x == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    
    // 每个线程处理一个元素对
    int idx = threadIdx.x;
    if (idx < 8) {
        // 将BF16转换为float进行计算
        float a_val = __bfloat162float(a[idx]);
        float b_val = __bfloat162float(b[idx]);
        atomicAdd(&shared_sum, a_val * b_val);
    }
    __syncthreads();
    
    // 第一个线程将结果加上c
    if (threadIdx.x == 0) {
        *result = shared_sum + c;
    }
}

// 执行八点积加计算（BF16版本）
cudaError_t computeDot8BF16(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b, float c, float& result) {
    // 准备设备内存
    __nv_bfloat16 *d_a, *d_b;
    float *d_result;
    
    cudaMalloc(&d_a, 8 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b, 8 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_a, a.data(), 8 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), 8 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // 启动内核
    dot8BF16Kernel<<<1, 8>>>(d_a, d_b, c, d_result);
    
    // 等待内核完成
    cudaError_t kernelStatus = cudaDeviceSynchronize();
    if (kernelStatus != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        return kernelStatus;
    }
    
    // 拷贝结果回主机
    cudaError_t copyStatus = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return copyStatus;
}

// 解析输入行
bool parseLine(const std::string& line, std::string& op, std::string& rounding,
               std::vector<__nv_bfloat16>& a, std::vector<__nv_bfloat16>& b, float& c) {
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
        a[i] = hexStringToBF16(tokens[2 + i]);
    }
    
    b.resize(8);
    for (int i = 0; i < 8; i++) {
        b[i] = hexStringToBF16(tokens[10 + i]);
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
        std::vector<__nv_bfloat16> a, b;
        float c;
        
        if (!parseLine(line, op, rounding, a, b, c)) {
            std::cerr << "解析第 " << lineNum << " 行时出错" << std::endl;
            continue;
        }
        
        if (op != "DOT8") {
            std::cerr << "第 " << lineNum << " 行不支持的操作: " << op << std::endl;
            continue;
        }
        
        if (rounding != "RND_NEAREST") {
            std::cerr << "第 " << lineNum << " 行不支持的舍入模式: " << rounding << std::endl;
            continue;
        }
        
        float result;
        cudaError_t status = computeDot8BF16(a, b, c, result);
        
        if (status != cudaSuccess) {
            std::cerr << "第 " << lineNum << " 行计算失败: " 
                      << cudaGetErrorString(status) << std::endl;
            continue;
        }
        
        // 写入结果到输出文件
        fout << line << ", " << floatToHexString(result) << std::endl;
    }
    
    fin.close();
    fout.close();
    
    std::cout << "处理完成。结果已写入 " << outputFile << std::endl;
    
    return 0;
}
