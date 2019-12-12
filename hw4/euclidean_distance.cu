#include<stdio.h>
#include<math.h>
#include<cuda_runtime.h>

#define MAX_RANGE 100
#define TILE_WIDTH 256


#define N = 1024;
#define M = 128;

#define funcCheck(stmt) do {                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        printf( "Failed to run stmt %d \n", __LINE__);                       \
        printf( "Got CUDA error ...  %s \n", cudaGetErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while(0)

void fullfill(float* A, int size) {
	for (int i = 0; i < size; i++) {
		A[i] = 0.0;
	}
}

__global__ void
gpuPdist(float *out, float *in, int n, int m) {
	extern __shared__ float Rs[256];
	float tmp, s;
	int myRow = blockIdx.x * 256 + threadIdx.x;
	for (int r = 0; r < n; r++) { //outer loop
		s = 0;
		for (int i = 0; i <= m / 256; i++) {
			if (i * 256 + threadIdx.x < m)
				Rs[i * 256 + threadIdx.x] = in[r*m +
				i * 256 + threadIdx.x];
		}
		__syncthreads();
		for (int i = 0; i < m && myRow < n; i++) {
			tmp = Rs[i] - in[myRow*m + i];
			s += tmp * tmp;
		}
		if (myRow < n)
			out[myRow*n + r] = sqrtf(s);
		__syncthreads();
	}
}

void cpu_distance_kernel(float* A, float* result, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = i + 1; j < M; j++) {
			float sum = 0;
			for (int k = 0; k < N; k++) {
				sum += (A[i*N + k] - A[j*N + k]) * (A[i*N + k] - A[j*N + k]);
			}
			// printf("%d %d\n",i,j);
			result[i*N + j] = (float)sqrt(sum);
		}
	}
}

int main()
{
	int total = M * N;
	int total_result = M * M;
	float *a, *result, *dev_a, *dev_result;
	float *tmp_result;


	a = (float*)malloc(sizeof(float)*total);
	result = (float*)malloc(sizeof(float)*total_result);
	tmp_result = (float*)malloc(sizeof(float)*total_result);
	funcCheck(cudaMalloc((void **)&dev_a, sizeof(float)*total));
	funcCheck(cudaMalloc((void **)&dev_result, sizeof(float)*total_result));

	// fullfill(a, total);
	fullfill(result, total_result);


	for (int i = 0; i < total; i++)
	{
		a[i] = (rand() % MAX_RANGE) / 2.0;
	}

	funcCheck(cudaMemcpy(dev_a, a, sizeof(float)*total_result, cudaMemcpyHostToDevice));

	int nGrid = total / TILE_WIDTH + 1;
	int nBlock = TILE_WIDTH;
	gpuPdist << <nGrid, nBlock >> > (dev_result, dev_a, M, N);

	cudaError_t err1 = cudaPeekAtLastError();
	cudaDeviceSynchronize();
	printf("Got CUDA error ... %s \n", cudaGetErrorString(err1));

	funcCheck(cudaMemcpy(tmp_result, dev_result, sizeof(float)*total_result, cudaMemcpyDeviceToHost));

	for (int i = 0; i < M; i++) {
		for (int j = i + 1; j < M; j++) {
			tmp_result[i*N + j] = sqrt(tmp_result[i*N + j]);
		}
	}

	printf("computing cpu kernel......\n");
	cpu_distance_kernel(a, result, M, N);

	printf("checking......\n");
	//bool checkFlag = true;
	//for (int i = 0; i < 2; i++) {
	//	for (int j = 0; j < N; j++) {
	//		printf("%d ", a[i*N + j]);
	//	}
	//	printf("\n");
	//}
	bool checkFlag = true;
	for (int i = 0; i < M; i++) {
		for (int j = i + 1; j < M; j++)
			if (tmp_result[i*N + j] != result[i*N + j]) {
				printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i, j, result[i], tmp_result[i]);
				checkFlag = false;
				break;
			}
		if (!checkFlag) {
			break;
		}
	}
	if (checkFlag) {
		printf("\tcheck pass.\n");
	}
	else {
		// printf("\tcheck fail.\n");
	}
	cudaFree(dev_a);
	cudaFree(dev_result);

	free(a);
	free(result);
	free(tmp_result);

	return 0;
}