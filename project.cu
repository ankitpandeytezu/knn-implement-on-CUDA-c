#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#define SIZE 128
   #define N 5
   #define M 5
   #define CHKSIZE 4

__global__ void EuclidianDistances( float *A, float *B , float *C , int n , int m)
{
        // SIZE is equal to 128
	__shared__ float accumResult[SIZE];
	__shared__ float sA[SIZE];
	__shared__ float sB[SIZE];

        // MAPPING
	int bx = blockIdx.x;  // n
	int by = blockIdx.y;  // m
	int ty = threadIdx.y; // 128
	int tx = threadIdx.x; // 1


	sA[ty] = A [bx * SIZE + ty];
	sB[ty] = B [by * SIZE + ty];
	__syncthreads();


	accumResult[ty] = (sA[ty] - sB[ty])*(sA[ty] - sB[ty]);
	__syncthreads();


	// Parallel tree-reduction
	for (int stride = SIZE/2 ; stride < 0 ; stride >>= 1)
		if (ty < stride)
			accumResult[ty]	+= accumResult [stride + ty];
	__syncthreads();

        // Writing results to output matrix
	if ((threadIdx.y == 0))
		C [bx * m + by] = accumResult[ty];
	__syncthreads();
}
 float comp_euclid_sq(const float *rA, const float *rB, const int size)
  {

  	 float result = 0.0f;
  	 float temp;
   	for (int i = 0; i < size; i++){
     		temp = ((rA[i]-rB[i])+(rA[i+1]-rB[i+1]));

     		if(temp<0)
     		temp=-temp;
     		result = temp;}
 //printf("%f",result);
   		return result;
   }  


int main()
{
      float et1=0.0f;//, et2=0.0f, et3=0.0f, et4=0.0f;
      cudaEvent_t start1, start2, start3,start4, stop1, stop2, stop3, stop4;
      cudaEventCreate(&start1);
      cudaEventCreate(&start2);
      cudaEventCreate(&start3);
      cudaEventCreate(&start4);
      cudaEventCreate(&stop1);
      cudaEventCreate(&stop2);
      cudaEventCreate(&stop3);
      cudaEventCreate(&stop4);

      int n = N;  //MatrixA size : n * SIZE
      int m = M; //MatrixB size : m * SIZE

      srand((unsigned)time(0));

      // Host Allocations
      float *matrixA = (float *) malloc (n * SIZE * sizeof(float));
      for(int i=0; i < n * SIZE; i++)
          matrixA[i] =(float) (rand()%100)+1;

      float *matrixB = (float *) malloc (m * SIZE * sizeof(float));
      for(int i=0; i < m * SIZE; i++)
          matrixB[i] = (float) (rand()%100)+1;

      float *results_kernel = (float *) malloc (n * m * sizeof(float));
      float *cpu_results_kernel = (float *) malloc (n * m * sizeof(float));
      for (int i = 0; i< n*m; i++)
        cpu_results_kernel[i] = comp_euclid_sq(matrixA + ((i/m)*SIZE), matrixB + (i%m)*SIZE, SIZE);

      //Device Allocation
      float *d_matrixA;
      float *d_matrixB;
      cudaMalloc((void **)&d_matrixA, n * SIZE * sizeof(float));
      cudaMalloc((void **)&d_matrixB, m * SIZE * sizeof(float));
       cudaMemcpy(d_matrixA , matrixA , n * SIZE * sizeof(float) , cudaMemcpyHostToDevice);
      cudaMemcpy(d_matrixB , matrixB , m * SIZE * sizeof(float) , cudaMemcpyHostToDevice);
      float *d_results_kernel;
      cudaMalloc((void **)&d_results_kernel , n * m * sizeof(float));


      dim3 threads1 (1 , SIZE);
      dim3 blocks1  (n , m);
      cudaEventRecord(start1);
      EuclidianDistances <<<blocks1 , threads1>>> (d_matrixA , d_matrixB , d_results_kernel , n , m);

      cudaEventRecord(stop1);
      cudaMemcpy(results_kernel , d_results_kernel , n * m *sizeof(float) , cudaMemcpyDeviceToHost);
      for (int i = 0; i< n*m; i++) 
       {
                  
        if (results_kernel[i] != cpu_results_kernel[i])  
         {
         printf("cpu/kernel1 mismatch at %d, cpu: %f, kernel1: %f\n", i, cpu_results_kernel[i], results_kernel[i]);
        }
       }
      
            cudaEventSynchronize(stop1);
            cudaEventElapsedTime(&et1, start1, stop1);
            printf("Element of Matrix A\n");
           for (int i = 0; i< n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                   printf("%.f\t",matrixA[(j*n)+i]);
                }
                   printf("\n");
                }
            printf("Element of Matrix B \n");
            for (int i = 0; i< n; i++)
            {
               for (int j = 0; j < m; j++)
               {
                  printf("%.f\t",matrixB[(j*n)+i]);
               }
                  printf("\n");
            }

         printf("distance matrix\n");
          for (int i = 0; i< n; i++)
         {
           for (int j = 0; j < m; j++)
           {
            printf("%.f\t",cpu_results_kernel[(j*n)+i]);
           }
         printf("\n");
        }

      printf("Success!\n");
      //printf("kernel1 : %.fms, kernel2 : %.fms, kernel3 : %.fms, kernel4 : %.fms\n", et1, et2, et3, et4);

      free(matrixA);
      free(matrixB);
 free(results_kernel);

     return 0;
                        
}
