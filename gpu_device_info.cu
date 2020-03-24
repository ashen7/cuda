#include "device_launch_parameters.h"
#include <iostream>
#include <string>

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; i++) {
        //cuda存放设备信息的结构体 
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        std::cout << "=============================================================================" << std::endl;
        std::cout << "使用GPU device：" << i << ": " << device_prop.name << std::endl;
        std::cout << "设备全局内存总量：" << device_prop.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM数量(一个线程块对应一个物理上的sm)：" << device_prop.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << device_prop.sharedMemPerBlock / 1024.0 << "KB" << std::endl;
        std::cout << "设备上一个线程块中可用的32位寄存器数量：" << device_prop.regsPerBlock << std::endl;
        std::cout << "每个SM的最大线程数：" << device_prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个SM的最大线程束数：" << device_prop.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量：" << device_prop.multiProcessorCount << std::endl; 
        std::cout << "=============================================================================" << std::endl;
    }

    return 0;
}
