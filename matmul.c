#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void transpose(double *A, double *B, int n) { // This function is to get the transpose of a matrix
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}

void mm(double *A, double *B, double *C, int n) // This is the matrix multiplication without transposing and without parallelizing
{   
    int i, j, k;
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            double dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B[k*n+j];
            } 
            C[i*n+j ] = dot;
        }
    }
}

void mm_omp(double *A, double *B, double *C, int n) // Parallelized matrix multiplication
{   
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                double dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B[k*n+j];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
}

void mmT(double *A, double *B, double *C, int n) // Matrix multiplication using transpose but not parallelized
{   
    int i, j, k;
    double *B2;
    B2 = (double*)malloc(sizeof(double)*n*n);
    transpose(B,B2, n);
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            double dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B2[j*n+k];
            } 
            C[i*n+j ] = dot;
        }
    }
    free(B2);
}

void mmT_omp(double *A, double *B, double *C, int n) // Matrix multiplication using transpose and parallelized
{   
    double *B2;
    B2 = (double*)malloc(sizeof(double)*n*n);
    transpose(B,B2, n);
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                double dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B2[j*n+k];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
    free(B2);
}

int main() {
    int i, n;
    double *A, *B, *C, dtime;
    int t;
    
    n=768;
    A = (double*)malloc(sizeof(double)*n*n);
    B = (double*)malloc(sizeof(double)*n*n);
    C = (double*)malloc(sizeof(double)*n*n);
    for(i=0; i<n*n; i++) { A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX;} // Generating random values for each of the matrices A and B
    printf("Set the number of threads: ");
    scanf("%d",&t);
    omp_set_num_threads(t);
    
    dtime = omp_get_wtime();
    mm(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("%f\n", dtime);

    dtime = omp_get_wtime();
    mm_omp(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("%f\n", dtime);

    dtime = omp_get_wtime();
    mmT(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("%f\n", dtime);

    dtime = omp_get_wtime();
    mmT_omp(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("%f\n", dtime);

    return 0;

}
