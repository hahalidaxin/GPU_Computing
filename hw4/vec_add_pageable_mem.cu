#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  //这里使用stride加循环的方法来处理数组大小不是block倍数的情况
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
    const unsigned int startTime = clock();

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    const int N = 1e8;
    size_t size = N * sizeof(float);

    float *a, *dev_a;
    float *b, *dev_b;
    float *c, *dev_c;

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

    //   cudaMemPrefetchAsync(a, size, deviceId);
    //   cudaMemPrefetchAsync(b, size, deviceId);
    //   cudaMemPrefetchAsync(c, size, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(dev_c, dev_a, dev_b, N);

    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    checkElementsAre(7, c, N);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    const unsigned int endTime = clock();

    printf("time cost : %d\n", endTime - startTime);
}