---
前言：
-
**因为要对一个矩阵进行优化加速，原有的openblas矩阵计算方法只是适合在做CPU端的加速，如果在线上有了GPU，这就使得使用GPU加速成为可能，并且也许会获得比较不错的性能结果。所以进行了尝试，进行矩阵的加速运算。**
* 如果图不清晰，请移至：https://blog.csdn.net/Binbin_Sun/article/details/80977237

---
第一部分：
-
* 相关背景和硬件信息介绍；使用的GPU为1080Ti，使用的cuda版本是8.0版本；驱动版本是384.111；相较于openblas需要自行编译，cublas 一般是在安装好cuda后就会有了；

* 关于行优先和列优先问题，正常的c/C++都是行优先，之前用过openblas，openblas默认的是列优先，不过openblas有设置为行优先的选项。（参见原有openblas文章的第一弹和第二弹）。现在使用的cublas是列优先的，并没有行列那个优先的设置选项；

---
第二部分：
-
* [cublas计算库](https://docs.nvidia.com/cuda/cublas/index.html)，包括：对矩阵自身的操作（点乘，求和，极大值和极小值等），矩阵与向量的计算,矩阵与矩阵的计算等；
```math
C = alpha * op(A) * op(B) +beta*C
```
以cublas中矩阵乘法作为示例，计算方法如下：

```c++
cublasStatus_t cublasSgemm(cublasHandle_t handle,//句柄，无含义
                           cublasOperation_t transa,//是否对A转置，即是否更换优先方式行/列
                           cublasOperation_t transb,//是否对B转置,即是否更换优先方式行/列
                           int m, int n, int k,
                           const float  *alpha,
                           const float  *A, int lda,//leading dimension of two-dimensional array used to store the matrix A.
                           const float  *B, int ldb,//leading dimension of two-dimensional array used to store the matrix B.
                           const float  *beta,
                           float *C, int ldc////leading dimension of two-dimensional array used to store the matrix C.
                           )
```
上面的接口参数中，比较好理解的是，m：这代表op(A)的行或是c的行；n，这个代表的是op（B）的列或是C的列，其中k，代表的是op（A）的列或是op（B）的行；其中alpha和beta如上面的公式可见，beta是修正偏差，只需要将alpha=1，beta=0，即可；比较难以理解的是lda，ldb，ldc这三个参数。参考api表示两维矩阵的leading dimension（主维度）；很奇怪为什么会需要这个选项，假设我要计算下面这样的矩阵乘法：
>$$A = [0,1,2,3,4,5,6,7,8,9]$$, $$B = [0,1,2,3,4,5]$$

>$$A= \begin{bmatrix}
   0& 1\\
    2 & 3 \\
   4 & 5\\
   6 & 7 \\
    8 & 9\\
  \end{bmatrix}_{5*2}*
  \begin{bmatrix}
    0& 1&2 \\
    3& 4 &5\\
      \end{bmatrix} _{2*3}\tag{1}$$

如果我们是要计算上面的结果，数据按照行排列，看起来是只需要给矩阵一个设定，告诉矩阵是$5*2$和$2*3$,就能够获得计算结果一个$5*3$的矩阵。


如下是代码cublas代码：
```c++
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>
int main(void){
    int const m = 5;
    int const n = 3;
    int const k = 2;
    float *A ,*B,*C;
    float *d_A,*d_B,*d_C;
    A = (float*)malloc(sizeof(float)*m*k);  //在内存中开辟空间
    B = (float*)malloc(sizeof(float)*n*k);  //在内存中开辟空间
    C = (float*)malloc(sizeof(float)*m*n); //在内存中开辟空间
    for(int i = 0; i< m*k; i++){
        A[i] = i;
        std::cout <<A[i]<<"\t";
    }
    std::cout <<"\n";
    for(int i = 0; i< n*k; i++){
        B[i] = i;
        std::cout <<B[i]<<"\t";
    }
    std::cout <<"\n";
    float alpha = 1.0;
    float beta = 0.0;
    cudaMalloc((void**)&d_A,sizeof(float)*m*k);
    cudaMalloc((void**)&d_B,sizeof(float)*n*k);
    cudaMalloc((void**)&d_C,sizeof(float)*m*n);
    for (int i = 0; i< m*n;i++){
        std::cout <<C[i]<<"\t";
    }
    std::cout <<"\n";
    cudaMemcpy(d_A,A,sizeof(float)*m*k,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*n*k,cudaMemcpyHostToDevice);
    for (int i = 0; i< m*k;i++){
        std::cout <<A[i]<<"\t";
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);//<测试一>
    //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, d_A, k, d_B, k, &beta, d_C, m);//<测试二>
    cudaMemcpy(C,d_C,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
    for (int i = 0; i< m*n;i++){
        std::cout <<C[i]<<"\t";
    }
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}
```

好了废话不多说了，直接说一下其中主维度的意思。因为cublas需要兼容Forthan等语言，使用了列优先的方式，并不像以前的openblas给出了选项，能够自行选择是行优先还是列优先。cublas默认的就是列优先，但是通过转置是能够转出行优先的效果的；


如果对A和B都选择了，不转置，那么相当于使用了默认的列排列，通过主维度lda，ldb，ldc确定了当选取了排列方式后，每列（行）排的数据。以A为例，列排列则先在每列排（lda = m = 5）个，然后同样的，B，选择列优先则先在每列排（ldb=k=2）个；
```c++
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
```

>$$A= \begin{bmatrix}
   0& 5\\
    1 & 6 \\
   2 & 7\\
   3 & 8 \\
    4 & 9\\
  \end{bmatrix}_{5*2}*
  \begin{bmatrix}
    0& 2&4 \\
    1& 3 &5\\
      \end{bmatrix} _{2*3}  
      =  \begin{bmatrix}
    5& 15&25 \\
    6& 20 &34\\
    7& 25&43\\
    8& 30 &52\\
    9& 35&61\\
      \end{bmatrix} _{5*3}  
\tag{测试一}$$

如果对A进行了转置，即相当于对A更换了行优先操作，那么相对应的lda也应该发生变化，（lda=m=5），变成（lda=k=2），如果不这样进行转换的话，那么就无法保证转换后的维度是对的。也就是如下的结果：
```
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, K, &alpha, d_A, k, d_B, k, &beta, d_C, m);
```
>$$A= \begin{bmatrix}
   0& 1\\
    2 & 3 \\
   4 & 5\\
   6 & 7 \\
    8 & 9\\
  \end{bmatrix}_{5*2}*
  \begin{bmatrix}
    0& 2&4 \\
    1& 3 &5\\
      \end{bmatrix} _{2*3}  
      =  \begin{bmatrix}
    1& 3&5 \\
    3& 13 &23\\
    5& 23&41\\
    7& 33 &59\\
    9& 43&77\\
      \end{bmatrix} _{5*3}  
\tag{测试二}$$

* 至此，关于矩阵计算问题就是这样了。需要根据自己选择来决定主维度的个数即lda，ldb，ldc；

---
第三部分：CPU和GPU的测试结果
-
* 如下是使用cublas和openblas的一些测试结果，仅供参考：
如下是149服务器上的测试结果：其中SGEMV=Matrix*vector，SGEMM = Matrix*Matrix，time_tocom表示比对次数；
GPU：cublas
SGEMV = 600000x512x1,         17.067844 s          time_tocom = 1000x
SGEMV = 1000000x512x1,         20.887469 s          time_tocom = 1000x
SGEMM = 1000000x512x1,         22.155032 s          time_tocom = 1000x
SGEMM = 1000000x512x5,         56.694733 s          time_tocom = 1000x
SGEMM = 4000000x512x5,         24.452547 s          time_tocom = 100x
CPU：openblas（其中openblas_set_num_threads(16)）
SGEMV = 600000x512x1,         69.089791 s          time_tocom = 1000x
SGEMV = 1000000x512x1,         134.489344 s          time_tocom = 1000x
SGEMM = 1000000x512x1,         220.625023 s          time_tocom = 1000x
SGEMM = 1000000x512x5,         282.610201 s          time_tocom = 1000x
SGEMM = 4000000x512x5,         100.310772 s          time_tocom = 100x
1：参考博客：说明的很好的博客，作者也做了一些有趣的测试：

https://blog.csdn.net/sinat_24143931/article/details/79487357

2： https://blog.csdn.net/u011197534/article/details/78378536