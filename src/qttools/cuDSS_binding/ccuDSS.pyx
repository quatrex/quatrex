import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from time import time

from libc.stdint cimport intptr_t, int64_t

cdef extern from "/usr/local/cuda-12.2/include/library_types.h":
    ctypedef enum cudaDataType_t:
        CUDA_R_16F
        CUDA_C_16F
        CUDA_R_16BF
        CUDA_C_16BF
        CUDA_R_32F
        CUDA_C_32F
        CUDA_R_64F
        CUDA_C_64F
        CUDA_R_4I
        CUDA_C_4I
        CUDA_R_4U
        CUDA_C_4U
        CUDA_R_8I
        CUDA_C_8I
        CUDA_R_8U
        CUDA_C_8U
        CUDA_R_16I
        CUDA_C_16I
        CUDA_R_16U
        CUDA_C_16U
        CUDA_R_32I
        CUDA_C_32I
        CUDA_R_32U
        CUDA_C_32U
        CUDA_R_64I
        CUDA_C_64I
        CUDA_R_64U
        CUDA_C_64U
        CUDA_R_8F_E4M3
        CUDA_R_8F_E5M2

cdef extern from "/home/mdossena/miniconda3/envs/testCUDSS/lib/python3.13/site-packages/nvidia/cu12/include/cudss.h":
    ctypedef struct cudssHandle_t:
        pass

    ctypedef struct cudssConfig_t:
        pass

    ctypedef struct cudssData_t:
        pass

    ctypedef struct cudssMatrix_t:
        pass

    ctypedef enum cudssStatus_t:
        CUDSS_STATUS_SUCCESS
        CUDSS_STATUS_NOT_INITIALIZED
        CUDSS_STATUS_ALLOC_FAILED
        CUDSS_STATUS_INVALID_VALUE
        CUDSS_STATUS_NOT_SUPPORTED
        CUDSS_STATUS_EXECUTION_FAILED
        CUDSS_STATUS_INTERNAL_ERROR
    
    ctypedef enum cudssLayout_t:
        CUDSS_LAYOUT_COL_MAJOR
        CUDSS_LAYOUT_ROW_MAJOR

    ctypedef enum cudssMatrixType_t:
        CUDSS_MTYPE_GENERAL
        CUDSS_MTYPE_SYMMETRIC
        CUDSS_MTYPE_HERMITIAN
        CUDSS_MTYPE_SPD
        CUDSS_MTYPE_HPD

    ctypedef enum cudssMatrixViewType_t:
        CUDSS_MVIEW_FULL
        CUDSS_MVIEW_LOWER
        CUDSS_MVIEW_UPPER

    ctypedef enum cudssIndexBase_t:
        CUDSS_BASE_ZERO
        CUDSS_BASE_ONE

    ctypedef enum cudssPhase_t:
        CUDSS_PHASE_ANALYSIS               
        CUDSS_PHASE_FACTORIZATION          
        CUDSS_PHASE_REFACTORIZATION     
        CUDSS_PHASE_SOLVE                 
        CUDSS_PHASE_SOLVE_FWD            
        CUDSS_PHASE_SOLVE_DIAG         
        CUDSS_PHASE_SOLVE_BWD          

    
    cudssStatus_t cudssSetThreadingLayer(cudssHandle_t handle, const char* thrLibFileName)
    cudssStatus_t cudssDataDestroy(cudssHandle_t handle, cudssData_t solverData)
    cudssStatus_t cudssCreate(cudssHandle_t *handle)
    cudssStatus_t cudssDestroy(cudssHandle_t handle)
    cudssStatus_t cudssConfigCreate(cudssConfig_t *solverConfig)
    cudssStatus_t cudssDataCreate(cudssHandle_t handle, cudssData_t *solverData)
    cudssStatus_t cudssMatrixCreateDn(cudssMatrix_t *matrix, int64_t nrows, int64_t ncols, int64_t ld, void *values, cudaDataType_t valueType,  cudssLayout_t layout)
    cudssStatus_t cudssMatrixCreateCsr(cudssMatrix_t *matrix, int64_t nrows, int64_t ncols, int64_t nnz, void *rowStart, void *rowEnd, void *colIndices, void *values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase)
    cudssStatus_t cudssExecute(cudssHandle_t handle, cudssPhase_t phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs)
    cudssStatus_t cudssMatrixDestroy(cudssMatrix_t matrix)
    cudssStatus_t cudssConfigDestroy(cudssConfig_t solverConfig)

def spsolve_with_CUDSS(A, b):
    cdef cudssStatus_t status
    cdef cudssHandle_t handle

    cdef int n = A.shape[0]
    cdef int nnz = A.nnz
    cdef int nrhs = b.shape[1]

    if A.dtype != cp.dtype(cp.complex128):
        print("The sys. matrix values should be complex128 (for now)")
        return -1
    
    if b.dtype != cp.dtype(cp.complex128):
        print("The rhs values should be complex128 (for now)")
        return -1

    x = cp.empty((n, nrhs), dtype=cp.complex128, order='F')

    cdef size_t  data_ptr_t = A.data.data.ptr
    cdef size_t  indices_ptr_t = A.indices.data.ptr
    cdef size_t  indptr_ptr_t = A.indptr.data.ptr
    cdef size_t  b_ptr_t = b.data.ptr
    cdef size_t  x_ptr_t = x.data.ptr

    cdef void * data_ptr = <void*>data_ptr_t
    cdef void * indices_ptr = <void*>indices_ptr_t
    cdef void * indptr_ptr = <void*>indptr_ptr_t
    cdef void * b_ptr = <void*>b_ptr_t
    cdef void * x_ptr = <void*>x_ptr_t


    status = cudssCreate(&handle)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Handle creation FAILED!",flush=True)
        return -1
    
    cdef cudssConfig_t solverConfig
    cdef cudssData_t solverData

    status = cudssSetThreadingLayer(handle, NULL)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Set threading layer FAILED!", flush=True)

    status = cudssConfigCreate(&solverConfig)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS SolverConfig creation FAILED!", flush=True)
        return -1

    status = cudssDataCreate(handle, &solverData)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS SolverData creation FAILED!",flush=True)
        return -1

    cdef cudssMatrix_t cu_x, cu_b
    cdef int64_t nrows = n, ncols = n
    cdef int ldb = ncols, ldx = nrows;

    status = cudssMatrixCreateDn(&cu_b, ncols, nrhs, ldb, b_ptr, CUDA_C_64F,
                         CUDSS_LAYOUT_COL_MAJOR)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Matrix b creation FAILED!",flush=True)
        return -1

    status = cudssMatrixCreateDn(&cu_x, nrows, nrhs, ldx, x_ptr, CUDA_C_64F,
                         CUDSS_LAYOUT_COL_MAJOR)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Matrix x creation FAILED!",flush=True)
        return -1

    cdef cudssMatrix_t cu_A
    cdef cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL
    cdef cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL
    cdef cudssIndexBase_t base       = CUDSS_BASE_ZERO

    status = cudssMatrixCreateCsr(&cu_A, nrows, ncols, nnz, indptr_ptr, NULL,
                         indices_ptr, data_ptr, CUDA_R_32I, CUDA_C_64F, mtype, mview,
                         base)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS CSR Matrix A creation FAILED!",flush=True)
        return -1

    cdef double t0, t1

    t0 = time()
    status =  cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                           cu_A, cu_x, cu_b)
    t1 = time()
    print(f"CUDSS Sym. Factorization took{' '*6}{t1 - t0:8.4f} seconds")
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Sym. Factorization FAILED!",flush=True)
        return -1

    t0 = time()
    status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                          solverData, cu_A, cu_x, cu_b)
    t1 = time()
    print(f"CUDSS Factorization took{' '*11}{t1 - t0:8.4f} seconds")
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Factorization FAILED!",flush=True)
        return -1

    t0 = time()
    status = cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                          cu_A, cu_x, cu_b)
    t1 = time()
    print(f"CUDSS Solve took{' '*19}{t1 - t0:8.4f} seconds")
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Solve FAILED!",flush=True)
        return -1

    status = cudssMatrixDestroy(cu_A)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS CSR Matrix A destruction FAILED!",flush=True)
        return -1
    
    status = cudssMatrixDestroy(cu_b)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Matrix b destruction FAILED!",flush=True)
        return -1
    
    status = cudssMatrixDestroy(cu_x)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Matrix x destruction FAILED!",flush=True)
        return -1
    
    status = cudssDataDestroy(handle, solverData)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Solver data destruction FAILED!",flush=True)
        return -1

    status = cudssConfigDestroy(solverConfig)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Solver config destruction FAILED!",flush=True)
        return -1

    status = cudssDestroy(handle)
    if status != CUDSS_STATUS_SUCCESS:
        print("CUDSS Handle destruction FAILED!",flush=True)
        
    
    return x

