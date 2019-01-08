#include<stdio.h>
#include<cuda_runtime.h>
/*#define CHECK(call)
{
        const cudaError_t error = call;
        if(error!= cudaSuccess)
        {
                 printf("error: %s:%d, ",__FILE__,__LINE__);
                 printf("code: %d, reason: %s\n",error, cudaGetErrorString(error));
                 exit(1);
        }
}*/

__global__ void hellowfromgpu(void)// GPU
{
         printf("hellow world from GPU\n");
}
 

//for CPU//
int main()
{
    printf("hellow from CPU\n");
    
    hellowfromgpu <<<1,10>>>();
    cudaDeviceReset();
    return 0;

}

