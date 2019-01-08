#include <stdio.h>

//#include "mex.h"

#include "cuda.h"



#define BLOCK_DIM 16





// Compute the distances between the matrix A and B.

// A   : reference points

// B   : query points

// wA  : theorical width of A

// wB  : theorical width of B

// pA  : real width of A (pitch in number of column)

// pB  : real width of B (pitch in number of column)

// dim : dimension = height of A and B

// C   : output matrix

__global__ void

cuComputeDistance( float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* C)

{

    // Block index

    int bx = blockIdx.x;

    int by = blockIdx.y;

    

    // Thread index

    int tx = threadIdx.x;

    int ty = threadIdx.y;

    

    // Sub-matrix of A : begin, step, end

    int begin_A = BLOCK_DIM * by;

    int end_A   = begin_A + (dim-1) * pA;

    int step_A  = BLOCK_DIM * pA;

    

    // Sub-matrix of B : begin, step

    int begin_B = BLOCK_DIM * bx;

    int step_B  = BLOCK_DIM * pB;

    

    // Csub is used to store the element of the block sub-matrix that is computed by the thread

    float Csub = 0;

    

    // Conditions

    int condA = (begin_A + ty < wA);

    int condB = (begin_B + tx < wB);

    

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix

    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        

        // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B

        __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];

        __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

        

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix

        if (a/pA + ty < dim){

            shared_A[ty][tx] = (begin_A + tx < wA)? A[a + pA * ty + tx] : 0;

            shared_B[ty][tx] = (begin_B + tx < wB)? B[b + pB * ty + tx] : 0;

        }

        else{

            shared_A[ty][tx] = 0;

            shared_B[ty][tx] = 0;

        }

        

        // Synchronize to make sure the matrices are loaded

        __syncthreads();

        

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix

        if (condA && condB){

            for (int k = 0; k < BLOCK_DIM; ++k)

                Csub += (shared_A[k][ty] -  shared_B[k][tx]) * (shared_A[k][ty] -  shared_B[k][tx]);

        }

        

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration

        __syncthreads();

    }

    

    // Write the block sub-matrix to device memory; each thread writes one element

    if (condA && condB){

        int c = (begin_A + ty)*pB + (begin_B + tx);

        C[c] = Csub;

    }

}









// Sort the column of matric tab.

// tab    : matrix containing distances

// width  : theorical width of tab

// height : height of tab

// pitch  : real width of tab (pitch in number of column)

__global__ void cuParallelCombSort(float *tab, int width, int height, int pitch){

    

    int     gap = height;

    float   swapped;

    float   temp;

    float * tab_tmp;

    int     index_1;

    int     index_2;

    int     i;

    

    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;

    

    if (xIndex<width){

        

        tab_tmp = &tab[xIndex];

        

        for (;;) {

            swapped = 0;

            

            gap = gap * 10 / 13;

            if (gap < 1)

                gap = 1;

            if (gap == 9 || gap == 10)

                gap = 11;

            

            

            for (i = 0; i < height - gap; i++) {

                index_1 = i*pitch;

                index_2 = (i+gap)*pitch;

                if (tab_tmp[index_1] > tab_tmp[index_2]) {

                    // swap

                    temp = tab_tmp[index_1];

                    tab_tmp[index_1] = tab_tmp[index_2];

                    tab_tmp[index_2] = temp;

                    swapped = 1;

                }

            }

            

            if (gap == 1 && !swapped)

                break;

        }

    }

}









// Get the k-th value of each column and compute its square root.

// dist   : matrix containing all the distances

// width  : theorical width of B

// pitch  : real width of A (pitch in number of column)

// k      : k-th nearest neighbor

// output : the output vector

__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k, float* output){

    unsigned int xIndex = blockIdx.x * BLOCK_DIM * BLOCK_DIM + threadIdx.x;

    if (xIndex<width)

        output[xIndex] = sqrt(dist[k*pitch+xIndex]);

}









// Print error message

// e        : error code

// var      : name of variable

// mem_free : size of free memory in bytes

// mem_size : size of memory tryed to be allocated

void printErrorMessageAndMatlabExit(cudaError_t e, char* var, int mem_free, int mem_size){

    printf("==================================================\n");

    printf("ERROR MEMORY ALLOCATION  : %s\n",cudaGetErrorString(e));

    printf("Variable                 : %s\n",var);

    printf("Free memory              : %d\n",mem_free);

    printf("Whished allocated memory : %d\n",mem_size);

    printf("==================================================\n");

    mexErrMsgTxt("CUDA ERROR DURING MEMORY ALLOCATION");

}









// MEX function

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    

    // Reference points

    float*      ref_host;

    float*      ref_dev;

    int         ref_width;

    int         ref_height;

    size_t      ref_pitch;

    cudaError_t ref_alloc;

    

    // Reference points

    float*      query_host;

    float*      query_dev;

    int         query_width;

    int         query_height;

    size_t     query_pitch;

    cudaError_t query_alloc;

    

    // Output array

    float*     output_host;

    float*     output_dev;

    size_t     output_pitch;

    cudaError_t output_alloc;

    

    // Distance

    float*     dist_dev;

    size_t     dist_pitch;

    cudaError_t dist_alloc;

    

    // K for K-Nearest Neighbor

    int k;

    

    // Size of float

    int sof = sizeof(float);

    

    // Memory information

    unsigned int mem_free, mem_total;

    

    // CUDA Init and get memory information

    cuInit(0);

    cuMemGetInfo(&mem_free,&mem_total);

    //printf("Free : %d\n",mem_free);

    

    // Read input information

    ref_host        = (float *) mxGetPr(prhs[0]);

    ref_width       = mxGetM(prhs[0]);

    ref_height      = mxGetN(prhs[0]);

    query_host      = (float *) mxGetPr(prhs[1]);

    query_width     = mxGetM(prhs[1]);

    query_height    = mxGetN(prhs[1]);

    k               = (int)mxGetScalar(prhs[2]);

    

    // Allocation of output array

    output_host = (float *) mxGetPr(plhs[0]=mxCreateNumericMatrix(query_width,1,mxSINGLE_CLASS,mxREAL));

    

    // Allocation CUDA memory

    cuMemGetInfo(&mem_free,&mem_total);

    dist_alloc = cudaMallocPitch( (void **) &dist_dev, &dist_pitch, query_width*sof, ref_width);

    if (dist_alloc)

    {

        // Print error message and matlab exit

        printErrorMessageAndMatlabExit(dist_alloc,"dist_dev", mem_free, query_width*ref_width*sof);

    }

     

    // Allocation CUDA memory

    cuMemGetInfo(&mem_free,&mem_total);

    ref_alloc = cudaMallocPitch( (void **) &ref_dev, &ref_pitch, ref_width*sof, ref_height);

    if (ref_alloc)

    {

        // Free memory of var already allocated

        cudaFree(dist_dev);        

        // Print error message and matlab exit

        printErrorMessageAndMatlabExit(ref_alloc,"dist_dev", mem_free, ref_width*ref_height*sof);

    }

     

    // Allocation CUDA memory

    cuMemGetInfo(&mem_free,&mem_total);

    query_alloc = cudaMallocPitch( (void **) &query_dev, &query_pitch, query_width*sof, query_height);

    if (query_alloc)

    {

        // Free memory of var already allocated

        cudaFree(dist_dev);

        cudaFree(ref_dev);

        // Print error message and matlab exit

        printErrorMessageAndMatlabExit(query_alloc,"dist_dev", mem_free, query_width*query_height*sof);

    }

     

    // Allocation CUDA memory

    cuMemGetInfo(&mem_free,&mem_total);

    output_alloc = cudaMallocPitch( (void **) &output_dev, &output_pitch, query_width*sof, 1);

    if (output_alloc)

    {

        // Free memory of var already allocated

        cudaFree(dist_dev);

        cudaFree(ref_dev);

        cudaFree(query_dev);

        // Print error message and matlab exit

        printErrorMessageAndMatlabExit(output_alloc,"dist_dev", mem_free, query_width*sof);

    }

     

    // Copy host to device

    cudaMemcpy2D(ref_dev   , ref_pitch    , ref_host    , ref_width*sof   ,  ref_width*sof  , ref_height    , cudaMemcpyHostToDevice);

    cudaMemcpy2D(query_dev , query_pitch  , query_host  , query_width*sof , query_width*sof , query_height  , cudaMemcpyHostToDevice);

    

    // Compute square distances

    dim3 grid(query_width/BLOCK_DIM, ref_width/BLOCK_DIM, 1);

    dim3 threads(BLOCK_DIM,BLOCK_DIM,1);

    if (query_width%BLOCK_DIM !=0) grid.x+=1;

    if (ref_width%BLOCK_DIM   !=0) grid.y+=1;

    cuComputeDistance<<<grid,threads>>>( ref_dev, ref_width, ref_pitch/sof, query_dev, query_width, query_pitch/sof, ref_height, dist_dev );



    // Sort each row in parallel

    dim3 grid2(query_width/BLOCK_DIM,1,1);

    dim3 threads2(BLOCK_DIM,1,1);

    if (query_width%BLOCK_DIM !=0) grid2.x+=1;

    cuParallelCombSort<<<grid2,threads2>>>(dist_dev, query_width, ref_width, dist_pitch/sof);



    // Compute sqrt

    dim3 grid3(query_width/(BLOCK_DIM*BLOCK_DIM),1,1);

    dim3 threads3(BLOCK_DIM*BLOCK_DIM,1,1);

    if (query_width%(BLOCK_DIM*BLOCK_DIM) !=0) grid3.x+=1;

    cuParallelSqrt<<<grid3,threads3>>>(dist_dev, query_width, query_pitch/sof, k-1, output_dev);

     

    // Copy memory

    cudaMemcpy2D(output_host, query_width*sof, output_dev, output_pitch, query_width*sof, 1, cudaMemcpyDeviceToHost);

    

    // Free memory

    cudaFree(dist_dev);

    cudaFree(ref_dev);

    cudaFree(query_dev);

    cudaFree(output_dev);

}


