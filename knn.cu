#include"cuda_runtime.h" 
#include"device_launch_parameters.h" 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <time.h> 
#include <math.h> 
#define row 500000
#define col 10 
#define rows 5000 
#define cols 10
 
int aba[row][col]; 
int aa[rows][cols]; 
double cc[rows][rows]; 
double cpu_out[rows][rows]; 
int k; 
int a, b, c, d;

double *d_c = 0; 
int *d_a = 0; 
int *d_i = 0;
 
__global__ void distance(const int * dev_a, double * dev_c, const int * dev_i) 
{ 
	int th_num = blockIdx.x * 128 + threadIdx.x; 
	double sum = 0; 
	for (int k = th_num + 1; k<rows; k++) 
	{	 
		for (int c = 0; c < cols; c++) 
		sum += (dev_a[th_num*cols + c] - dev_i[k*cols + c])*(dev_a[th_num*cols + c] - dev_i[k*cols + c]); 
		dev_c[th_num] = sqrt(sum); 
	} 
// printf ("Sum: %d \t " , sum); 
} 
__global__ void sorting(double *dev_c, double *sort, double K) 
{
 
	int temp; int i; 
	i = blockIdx.x * 128 + threadIdx.x; 
	for (int r = 0; r < rows; r++) 
	{ 
		for (int c = 0; c < cols; c++) 
		{ 
			if (dev_c[cols*r + c]<dev_c[cols* i + c]) 
			{ 
				temp = dev_c[cols*r + c]; 
				dev_c[cols*r + c] = dev_c[cols* i + c]; 
				dev_c[cols* i + c] = temp; 
			} 
		} 
	} 
} 

int main() 
{ 
printf(" enter the number of K nearest neighbors :"); 
scanf("%d", &k); 
FILE *myFile; 
myFile = fopen("name.csv ", "w+"); 
if (myFile == NULL) 
{ 
	printf("Error Reading File\n"); 
	exit(0); 
} 
char buffer[1024];
int i = 0, j = 0; 
char *record, *line; 
while ((line = fgets(buffer, sizeof(buffer), myFile)) != NULL) 
{ 
	j = 0; 
	record = strtok(line, " ,"); 
	while (record != NULL) 
	{ 
// printf("%d \t %d \t %d \n" , ( cols * i ) + j , i , j ); 
		aba[i][j] = atoi(record); 
		record = strtok(NULL, " ,"); 
		j++; 
	} 
	i++; 
}
 
fclose(myFile); 
int input[cols]; 
for (int i = 0; i <cols; i++) 
{ 
	int x; 
	printf("enter input %d\n", i); 
	scanf("%f\n", &input[i]); 

	if (input[i] == 1001001) 
	{ 
		for (x = 0; x < row; x++) 
		{ 
			aba[x][i] = 1001001; 
		} 
	} 
}
 
for (i = 0; i < cols; i++) 
{ 
	printf("%d", input[i]); 
} 
clock_t start; start = clock(); 
for (a = 0; a < 199; a++) 
{ 
	for (b = 0; b <5000; b++) 
	{ 
		c = a * 5000 + b; 
		for (d = 0; d <cols; d++) 
		{ 
			aa[b][d] = aba[c][d]; 
printf(" %d\n", aa[b][d]); 
		} 
scanf("%d", &k); 
	} 
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset(); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaDeviceReset failed !"); 
		return 1; 
	} 
	cudaStatus = cudaSetDevice(0); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaSetDevice failed ! Do you have a CUDAâˆ’capable GPU installed ?"); 
		goto Error; 
	}
 
	else printf("Working \n"); 
		cudaStatus = cudaMalloc((void **)&d_a, rows*cols * sizeof(int)); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaMalloc failed !"); goto Error; 
	} 
	else 
		printf(" Success ! ! ! \n"); 
		cudaStatus = cudaMalloc((void **)&d_i, cols * sizeof(int)); 

	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaMalloc failed !"); goto Error; 
	} 
	else 
		printf(" Success ! ! ! \n"); 
	cudaStatus = cudaMemcpy(d_i, input, cols * sizeof(int *), cudaMemcpyHostToDevice); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaMemcpy failed !"); 
		goto Error; 
	} 
	else printf(" Success ! ! ! \n"); 
	cudaStatus = cudaMemcpy(d_a, aa, rows*cols * sizeof(int *), cudaMemcpyHostToDevice); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaMemcpy failed !"); 
		goto Error; 
	} 
	else printf(" Success ! ! ! \n"); 
	cudaStatus = cudaMalloc((void **)&d_c, rows* rows * sizeof(double)); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaMalloc failed !"); 
		goto Error; 
	} 
	else printf(" Success ! ! ! \n"); 
	double *sort = 0; 
	cudaStatus = cudaMalloc((void **)&sort, rows* rows * sizeof(double)); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaMalloc failed !"); 
		goto Error; 
	} 
	else 
		printf(" Success ! ! ! \n"); 
	int threads = 128; 
	while (rows%threads != 0) threads++; 
		printf("TH: %d \n", threads); 
//return 0; 
	dim3 threadsPerBlock(threads); 
	dim3 numBlocks(rows / threadsPerBlock.x); 
	distance << <numBlocks, threadsPerBlock >> > (d_a, d_c, d_i); 
	sorting << <numBlocks, threadsPerBlock >> > (d_c, sort, k); 
	cudaStatus = cudaGetLastError(); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "addKern launch failed : %s\n", cudaGetErrorString(cudaStatus)); 
		goto Error; 
	} 
	cudaStatus = cudaDeviceSynchronize(); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel !\n", cudaStatus); 
		scanf("%d", &k); 
		goto Error; 
	} 
//return cudaStatus ; 
	cudaStatus = cudaMemcpy(cc, d_c, rows*rows * sizeof(double), cudaMemcpyDeviceToHost); 
	if (cudaStatus != cudaSuccess) 
	{ 
		fprintf(stderr, "addKernel launch failed : %s\n", cudaGetErrorString(cudaStatus)); 
		goto Error; 
	} 
}
 
printf("GPU Time Taken: %f \n", (double)(clock() - start) / CLOCKS_PER_SEC); 
scanf("%d", &k); 
for (int l = 0; l <= k; l++) 
{ 
 for (i = 0; i < rows; i++) 
 { 
 for (int j = 0; j < rows; j++) 
 { 
 printf("%f \t ", cc[(rows * i) + j]); 
 } 
 } 
} 
Error: 
printf (" Exiting . . \n"); 
scanf("%d", &k); 
cudaFree(d_c); 
cudaFree(d_a); 
cudaFree(d_i); 
} 


