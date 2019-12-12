#include <stdio.h>

//此处N不是block的整数倍
#define N  1024

__global__ void matrixMulGPU( int * a, int * b, int * result )
{
  int val = 0;

  //使用stride加循环处理不是整数倍的情况
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int strideX = blockDim.x * gridDim.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int strideY = blockDim.y * gridDim.y;

  for(int r = row; r<N;r+=strideX)
  for (int c = col;c<N;c+=strideY)
  {
    for ( int k = 0; k < N; ++k )
      val += a[r * N + k] * b[k * N + c];
    result[r * N + c] = val;
  }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory; create 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */

  dim3 threads_per_block (32, 32, 1); // A 16 x 16 block threads
  dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

  const unsigned int startTime = clock();
  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaDeviceSynchronize();

  const unsigned int endTime = clock();
  printf("calc mat multi using : %d\n", endTime-startTime);

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row ) {
    for( int col = 0; col < N && !error; ++col ) {
      //printf("%d ",c_gpu[row * N + col]);
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
      }
      //printf("\n");
  }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}
