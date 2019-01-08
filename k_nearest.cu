 #include<stdio.h>
  2 #include<cuda.h>
  3 #include<cuda_runtime.h>
  4 #define SIZE 128
  5    #define N 5
  6    #define M 5
  7    #define CHKSIZE 4
  8
  9 __global__ void EuclidianDistances( float *A, float *B , float     *C , int n , int m)
 10 {
 11         // SIZE is equal to 128
 12         __shared__ float accumResult[SIZE];
 13         __shared__ float sA[SIZE];
 14         __shared__ float sB[SIZE];
 // MAPPING
 17         int bx = blockIdx.x;  // n
 18         int by = blockIdx.y;  // m
 19         int ty = threadIdx.y; // 128
 20         int tx = threadIdx.x; // 1
 21
 22
 23         sA[ty] = A [bx * SIZE + ty];
 24         sB[ty] = B [by * SIZE + ty];
 25         __syncthreads();
 26
 27
 28         accumResult[ty] = (sA[ty] - sB[ty])*(sA[ty] - sB[ty]);
 29         __syncthreads();
 30

 32         // Parallel tree-reduction
 33         for (int stride = SIZE/2 ; stride < 0 ; stride >>= 1)
 34                 if (ty < stride)
 35                         accumResult[ty] += accumResult [stride     + ty];
 36         __syncthreads();
 37
 38         // Writing results to output matrix
 39         if ((threadIdx.y == 0))
 40                 C [bx * m + by] = accumResult[ty];
 41         __syncthreads();
                                  }
 43  float comp_euclid_sq(const float *rA, const float *rB, const     int size)
 44   {
 45
 46          float result = 0.0f;
 47          float temp;
 48         for (int i = 0; i < size; i++){
 49                 temp = ((rA[i]-rB[i])+(rA[i+1]-rB[i+1]));
 50
 51                 if(temp<0)
  temp=-temp;
 53                 result = temp;}
 54  //printf("%f",result);
 55                 return result;
 56    }
 57
 58
 59 int main()
 60 {
 61       float et1=0.0f;//, et2=0.0f, et3=0.0f, et4=0.0f;
 62       cudaEvent_t start1, start2, start3,start4, stop1, stop2,     stop3, stop4;
 63       cudaEventCreate(&start1);
 64       cudaEventCreate(&start2);
 65       cudaEventCreate(&start3);
 cudaEventCreate(&start4);
 67       cudaEventCreate(&stop1);
 68       cudaEventCreate(&stop2);
 69       cudaEventCreate(&stop3);
 70       cudaEventCreate(&stop4);
 71
 72       int n = N;  //MatrixA size : n * SIZE
 73       int m = M; //MatrixB size : m * SIZE
 74
 75       srand((unsigned)time(0));
 76
 77       // Host Allocations
 78       float *matrixA = (float *) malloc (n * SIZE * sizeof(flo    at));
 for(int i=0; i < n * SIZE; i++)
 80           matrixA[i] =(float) (rand()%100)+1;
 81
 82       float *matrixB = (float *) malloc (m * SIZE * sizeof(flo    at));
 83       for(int i=0; i < m * SIZE; i++)
 84           matrixB[i] = (float) (rand()%100)+1;
 85
 86       float *results_kernel = (float *) malloc (n * m * sizeof    (float));
 87       float *cpu_results_kernel = (float *) malloc (n * m * si    zeof(float));
 88       for (int i = 0; i< n*m; i++)
 cpu_results_kernel[i] = comp_euclid_sq(matrixA + ((i/m    )*SIZE), matrixB + (i%m)*SIZE, SIZE);
 90
 91       //Device Allocation
 92       float *d_matrixA;
 93       float *d_matrixB;
 94       cudaMalloc((void **)&d_matrixA, n * SIZE * sizeof(float)    );
 95       cudaMalloc((void **)&d_matrixB, m * SIZE * sizeof(float)    );
 96        cudaMemcpy(d_matrixA , matrixA , n * SIZE * sizeof(floa    t) , cudaMemcpyHostToDevice);
 float *d_results_kernel;
 99       cudaMalloc((void **)&d_results_kernel , n * m * sizeof(f    loat));
100
101
102       dim3 threads1 (1 , SIZE);
103       dim3 blocks1  (n , m);
104       cudaEventRecord(start1);
105       EuclidianDistances <<<blocks1 , threads1>>> (d_matrixA ,     d_matrixB , d_results_kernel , n , m);

107       cudaEventRecord(stop1);
108       cudaMemcpy(results_kernel , d_results_kernel , n * m *si    zeof(float) , cudaMemcpyDeviceToHost);
109       for (int i = 0; i< n*m; i++)
110        {
111
112         if (results_kernel[i] != cpu_results_kernel[i])
113          {
114          printf("cpu/kernel1 mismatch at %d, cpu: %f, kernel1:     %f\n", i, cpu_results_kernel[i], results_kernel[i]);
115         }
116        }
                                   cudaEventSynchronize(stop1);
119             cudaEventElapsedTime(&et1, start1, stop1);
120             printf("Element of Matrix A\n");
121            for (int i = 0; i< n; i++)
122             {
123                 for (int j = 0; j < m; j++)
124                 {
125                    printf("%.f\t",matrixA[(j*n)+i]);
126                 }
127                    printf("\n");
128                 }
129             printf("Element of Matrix B \n");
130             for (int i = 0; i< n; i++)
131             {
132                for (int j = 0; j < m; j++)
 {
134                   printf("%.f\t",matrixB[(j*n)+i]);
135                }
136                   printf("\n");
137             }
138
139          printf("distance matrix\n");
140           for (int i = 0; i< n; i++)
141          {
142            for (int j = 0; j < m; j++)
143            {
144             printf("%.f\t",cpu_results_kernel[(j*n)+i]);
145            }
146          printf("\n");
147         }
148
printf("Success!\n");
150       //printf("kernel1 : %.fms, kernel2 : %.fms, kernel3 : %.    fms, kernel4 : %.fms\n", et1, et2, et3, et4);
151
152       free(matrixA);
153       free(matrixB);
154  free(results_kernel);
155
156      return 0;
157
158 }


