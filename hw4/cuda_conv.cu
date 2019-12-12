#include<stdio.h>

__global__ void convolutionGPU(
                                float *d_Result,
                                float *d_Data,
                                float *d_Kernel,
                                int dataW,
                                int dataH )
{
    const int gLoc = threadIdx.x +
                        blockIdx.x * blockDim.x +
                        threadIdx.y * dataW +
                        blockIdx.y * blockDim.y * dataW;  
    float sum = 0;
    float value = 0;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) 
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) 
        {
            if (blockIdx.x == 0 && (threadIdx.x + i) < 0) 
                value = 0;
            else if ( blockIdx.x == (gridDim.x - 1) &&
                    threadIdx.x + i) > blockDim.x-1 ) 
                value = 0;
            else
            {
                if (blockIdx.y == 0 && (threadIdx.y + j) < 0)
                    value = 0;
                else if ( blockIdx.y == (gridDim.y - 1) &&
                        (threadIdx.y + j) > blockDim.y-1 )
                    value = 0;
                else 
                    value = d_Data[gLoc + i + j * dataW];
            }
            sum += value * d_Kernel[KERNEL_RADIUS + i] * d_Kernel[KERNEL_RADIUS + j];
        }
        d_Result[gLoc] = sum;
}

int main()
{
    
}