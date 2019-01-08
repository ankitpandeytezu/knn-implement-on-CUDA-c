#include<cuda_runtime.h>
#include<stdio.h>

__global__ void checkIndex(void)
{
        printf("threadIdx: (%d %d %d) blockIdx: (%d %d %d) blockDim: (%d %d %d) "
                " gridDim: (%d %d %d)\n",threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,                  blockDim.z,gridDim.x,gridDim.y,gridDim.z);  
}
  

int main(int args, char **argv)
{
    int n=6;
    // define grid and block structure//

    dim3 block(3);
    dim3 grid((n+block.x-1)/block.x);
    
// ckeck grid and block dimension from host side//
    printf("grid.x %d grid.y %d grid.x %d\n",grid.x,grid.y,grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
// check grid and block from device side//

    checkIndex <<<grid, block>>> ();
    cudaDeviceReset();
    return 0;

}
