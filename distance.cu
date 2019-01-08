#include <cuda_runtime.h>
#include <stdio.h>
#include<math.h>

/*#define CHECK(call) 
{ 
	const cudaError_t error = call; 
	if (error != cudaSuccess) 
	{ 
		printf("Error: %s:%d, ", __FILE__, __LINE__);
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
		exit(-10*error);
	}
}*/

void initialInt(int *ip, int size) 
{
	for (int i=0; i<size; i++)
	{
		ip[i] = i;
	}
}

void printMatrix(int *C, const int nx, const int ny)
{
	int *ic = C;
	printf("\nMatrix: (%d.%d)\n",nx,ny);
	for (int iy=0; iy<ny; iy++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			printf("%3d",ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy*nx + ix;
	printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index %2d ival %2d\n", threadIdx.x, threadIdx.y,
		blockIdx.x,blockIdx.y, ix, iy, idx, A[idx]);
}
__global__ void cudaEuclid( int* A, int* B, int* C, int nx, int ny )
    {
      //  int  squareeucldist = 0;
        int r = blockDim.x * blockIdx.x + threadIdx.x; // rows
        int c = blockDim.y * blockIdx.y + threadIdx.y; // cols 
        extern __shared__ float sdata[];
        //int r = blockIdx.y; int c = threadIdx.x;
        if( r < nx && c < ny  ){

            //C[r + rows*c] = ( A[r + rows*c] - B[r + rows*c] ) * ( A[r + rows*c] - B[r + rows*c] );


            sdata[threadIdx.x] = ( A[r + nx*c] - B[r + nx*c] ) * ( A[r + nx*c] - B[r + nx*c] );

            __syncthreads();

            // contiguous range pattern
            for(int offset = blockDim.x / 2;
                offset > 0;
                offset >>= 1)
            {
                if(threadIdx.x < offset)
                {
                    // add a partial sum upstream to our own
                    sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                }

                // wait until all threads in the block have
                // updated their partial sums
                __syncthreads();
            }

            // thread 0 writes the final result
            if(threadIdx.x == 0)
            {
                C[r] = sdata[0];
            }

        }

    }


int main(int argc, char **argv)
{
       int *A,*B,*C;
	printf("%s Starting...\n", argv[0]);
	// get device information
	int dev = 0;
	cudaDeviceProp deviceProp;
//	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
//	CHECK(cudaSetDevice(dev));

	// set matrix dimension
	int nx = 5;
	int ny = 5;
	int nxy = nx*ny;
	int nBytes = nxy * sizeof(float);

	// malloc host memory
	int *h_A;
	h_A = (int *)malloc(nBytes);

	int *h_B;
        h_B = (int *)malloc(nBytes);
        initialInt(h_B, nxy);
         cudaMalloc((void **)&B, nBytes);

         int *h_C;
         h_C = (int *)malloc(nBytes);
         initialInt(h_C, nxy);
          cudaMalloc((void **)&C, nBytes);

// iniitialize host matrix with integer
	initialInt(h_A, nxy);
	printMatrix(h_A, nx, ny);

	// malloc device memory
	
	cudaMalloc((void **)&A, nBytes);

 cudaMalloc((void **)&B, nBytes);
 cudaMalloc((void **)&C, nBytes);


	// transfer data from host to device
	cudaMemcpy(A, h_A, nBytes, cudaMemcpyHostToDevice);

cudaMemcpy(B, h_B, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(C, h_C, nBytes, cudaMemcpyHostToDevice);

	// set up execution configuration
	dim3 block(4, 2);
	dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
       // cudaEuclid( float* A, float* B, float* C, int rows, int cols );


	// invoke the kernel
//	printThreadIndex <<< grid, block >>>(A, nx, ny);
        dim3 dimBlock( ny, 1 ); 
        dim3 dimGrid( 1, nx ); 
        cudaEuclid<<<dimGrid, ny, ny*sizeof(float)>>>( A, B, C, nx, ny );
        
	cudaDeviceSynchronize();

	// free host and devide memory
 printMatrix(C, nx, ny);
  


	cudaFree(A);
	free(h_A);
        

	// reset device
	cudaDeviceReset();
	return (0);
}
