#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t size_x, size_t size_y) {
    CudaDims dim;
    size_t num_blocks_x = (size_x + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    size_t num_blocks_y = (size_y + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    dim.block = dim3(BASE_THREAD_NUM, BASE_THREAD_NUM, 1);
    dim.grid = dim3(num_blocks_x, num_blocks_y, 1);
    return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

struct CudaVecSigned {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

struct CudaArrayVec {
    uint32_t size;
    scalar_t* data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

CudaVecSigned VecToCuda(const std::vector<int32_t>& x) {
  CudaVecSigned shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

CudaArrayVec ArrayVecToCuda(const std::vector<CudaArray*>& x) {
    CudaArrayVec arrays;
    if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
    arrays.size = x.size();
    for (size_t i = 0; i < x.size(); i++) {
    arrays.data[i] = x[i]->ptr;
    }
    return arrays;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVecSigned shape,
                              CudaVecSigned strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t cur_stride = size;
        size_t _offset = 0;
        size_t data_offset = gid;
        for (size_t i = 0; i < shape.size; i++) {
            cur_stride /= shape.data[i];
            size_t s = data_offset / cur_stride;
            _offset += s * strides.data[i];
            data_offset = data_offset - s * cur_stride;
        }
        const scalar_t* ptr = a + offset;
        out[gid] = ptr[_offset];
    }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
    CudaDims dim = CudaOneDim(out->size);
    CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        uint32_t cur_stride = size;
        uint32_t _offset = 0;
        uint32_t data_offset = gid;
        for (size_t i = 0; i < shape.size; i++) {
            cur_stride /= shape.data[i];
            uint32_t s = data_offset / cur_stride;
            _offset += s * strides.data[i];
            data_offset = data_offset - s * cur_stride;
        }
        scalar_t* ptr = out + offset;
        ptr[_offset] = a[gid];
    }
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
    size_t size = 1;
    for (uint32_t s : shape) {
        size *= s;
    }
    CudaDims dim = CudaOneDim(size);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t cur_stride = size;
        size_t _offset = 0;
        size_t data_offset = gid;
        for (size_t i = 0; i < shape.size; i++) {
            cur_stride /= shape.data[i];
            size_t s = data_offset / cur_stride;
            _offset += s * strides.data[i];
            data_offset = data_offset - s * cur_stride;
        }
        scalar_t* ptr = out + offset;
        ptr[_offset] = val;
    }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
    CudaDims dim = CudaOneDim(size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape), VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void SparseDenseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, scalar_t* row, scalar_t* col, size_t size, uint32_t M, uint32_t N) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t x = gid / N;
        size_t y = gid - x * N;
        scalar_t val = b[gid];
        for (size_t nn = static_cast<size_t>(row[x]); nn < static_cast<size_t>(row[x+1]); nn++) {
            size_t idx = static_cast<size_t>(col[nn]);
            if (idx == y){
                val += a[nn];
                break;
            }   
        }
        out[gid] = val;
    }
}
void SparseDenseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out, const CudaArray& row, const CudaArray& col, uint32_t M, uint32_t N) {
    CudaDims dim = CudaOneDim(out->size);
    SparseDenseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, row.ptr, col.ptr, out->size, M, N);
}


__global__ void SparseDenseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, scalar_t* row, scalar_t* col, size_t size, uint32_t M, uint32_t N) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t x = gid / N;
        size_t y = gid - x * N;
        scalar_t val = b[gid];
        bool found = false;
        for (size_t nn = static_cast<size_t>(row[x]); nn < static_cast<size_t>(row[x+1]); nn++) {
            size_t idx = static_cast<size_t>(col[nn]);
            if (idx == y){
                val *= a[nn];
                found = true;
                break;
            }   
        }
        found? out[gid] = val : out[gid] = 0;
    }
}
void EwiseSparseDenseMul(const CudaArray& a, const CudaArray& b, CudaArray* out, const CudaArray& row, const CudaArray& col, uint32_t M, uint32_t N) {
    CudaDims dim = CudaOneDim(out->size);
    SparseDenseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, row.ptr, col.ptr, out->size, M, N);
}


__global__ void SparseDenseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, scalar_t* row, scalar_t* col, size_t size, uint32_t M, uint32_t N) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t x = gid / N;
        size_t y = gid - x * N;
        scalar_t val = b[gid];
        bool found = false;
        for (size_t nn = static_cast<size_t>(row[x]); nn < static_cast<size_t>(row[x+1]); nn++) {
            size_t idx = static_cast<size_t>(col[nn]);
            if (idx == y){
                val = a[nn]/val;
                found = true;
                break;
            }   
        }
        found? out[gid] = val : out[gid] = 0;
    }
}
void EwiseSparseDenseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out, const CudaArray& row, const CudaArray& col, uint32_t M, uint32_t N) {
    CudaDims dim = CudaOneDim(out->size);
    SparseDenseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, row.ptr, col.ptr, out->size, M, N);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}
void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void SparseScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, scalar_t* row, scalar_t* col, size_t size, uint32_t M, uint32_t N) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t x = gid / N;
        size_t y = gid - x * N;
        scalar_t out_val = val;
        for (size_t nn = static_cast<size_t>(row[x]); nn < static_cast<size_t>(row[x+1]); nn++) {
            size_t idx = static_cast<size_t>(col[nn]);
            if (idx == y){
                out_val += a[nn];
                break;
            }   
        }
        out[gid] = out_val;
    }
}
void SparseScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out, const CudaArray& row, const CudaArray& col, uint32_t M, uint32_t N) {
    CudaDims dim = CudaOneDim(out->size);
    SparseScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, row.ptr, col.ptr, out->size, M, N);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Mul together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    if (a[gid] == 0.0 && b[gid] == 0.0) {
        out[gid] = 0.0;
    } else {
        out[gid] = a[gid] / b[gid];
    }
  }
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Div together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Div together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Power together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] > b[gid] ? a[gid] : b[gid];
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] > val ? a[gid] : val;
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Mul together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == b[gid];
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == val;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Mul together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= b[gid];
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= val;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Mul together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseReluKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
      if (a[gid] >= 0.0) out[gid] = a[gid];
      else out[gid] = 0.0;
  }
}

void EwiseRelu(const CudaArray& a, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseReluKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseSigmoidKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = 1.0 / (1.0 + exp(-a[gid]));
  }
}

void EwiseSigmoid(const CudaArray& a, CudaArray* out) {
  /**
   * Mul together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseSigmoidKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = gid / P;
    size_t col = gid % P;
    if (row < M && col < P) {
        scalar_t sum = 0;
        for (size_t i = 0; i < N; ++i) {
        sum += a[row * N + i] * b[i * P + col];
        }
        out[gid] = sum;
    }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

    CudaDims dim = CudaOneDim(out->size);
    MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

__global__ void SparseDenseMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, scalar_t* row, scalar_t* col, uint32_t M, uint32_t N, uint32_t P) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < M * P) {
        size_t x = gid / P;
        size_t y = gid - x * P;
        scalar_t val = 0.0;

        for (size_t nn = static_cast<size_t>(row[x]); nn < static_cast<size_t>(row[x+1]); nn++) {
            size_t idx = static_cast<size_t>(col[nn]);
            scalar_t _a = a[nn];
            scalar_t _b = b[idx * P + y];
            val += _a * _b;
        }
        out[x * P + y] = val;
    }
}
void SparseDenseMatmul(const CudaArray& a, const CudaArray& b, CudaArray* out, const CudaArray& row, const CudaArray& col, uint32_t M, uint32_t N,
            uint32_t P) {
    CudaDims dim = CudaOneDim(M * P);
    SparseDenseMatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, row.ptr, col.ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t start_idx = gid * reduce_size;
        scalar_t cur_max = a[start_idx];
        for (size_t i = start_idx; i < start_idx + reduce_size; i++) {
            scalar_t val = a[i];
            if (val > cur_max) cur_max = val;
        }
        out[gid] = cur_max;
    }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    CudaDims dim = CudaOneDim(out->size);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t start_idx = gid * reduce_size;
        scalar_t cur_sum = 0.0;
        for (size_t i = start_idx; i < start_idx + reduce_size; i++) {
            cur_sum += a[i];
        }
        out[gid] = cur_sum;
    }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    CudaDims dim = CudaOneDim(out->size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}

__global__ void StackKernel(const CudaArrayVec arrays, scalar_t* out, CudaVec shape, CudaVec strides, CudaVec new_shape, size_t axis, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        uint32_t cur_stride = size;
        uint32_t _offset = 0;
        uint32_t data_offset = gid;
        uint32_t stride_offset = 0;
        uint32_t array_idx;
        for (size_t i = 0; i < new_shape.size; i++) {
            cur_stride /= new_shape.data[i];
            uint32_t s = data_offset / cur_stride;
            if (i != axis) {
                _offset += s * strides.data[stride_offset];
                ++stride_offset;
            } else {
                array_idx = s;
            }
            data_offset = data_offset - s * cur_stride;
        }
        out[gid] = arrays.data[array_idx][_offset];
    }
}

void Stack(const std::vector<CudaArray*>& arrays, CudaArray* out, std::vector<uint32_t> shape, std::vector<uint32_t> strides, std::vector<uint32_t> new_shape, uint32_t axis) {
    CudaDims dim = CudaOneDim(out->size);
    StackKernel<<<dim.grid, dim.block>>>(ArrayVecToCuda(arrays), out->ptr, VecToCuda(shape), VecToCuda(strides), VecToCuda(new_shape), axis, out->size);
}



__global__ void SwapKernel(const scalar_t* a, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t offset = size / 2;
        if (gid < offset) {
            out[gid] = a[gid+offset];
        } else {
            out[gid] = a[gid-offset];
        }
    }
}

void Swap(const CudaArray& a, CudaArray* out) {
    CudaDims dim = CudaOneDim(out->size);
    SwapKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void ToSparseByIndicesKernel(const scalar_t* a, scalar_t* out, CudaVecSigned strides, const scalar_t* indices, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t offset = indices[gid] * strides.data[0] + indices[gid+size] * strides.data[1];
        out[gid] = a[offset];
    }
}

void ToSparseByIndices(const CudaArray& a, CudaArray* out, std::vector<int32_t> strides, const CudaArray& indices) {
    Fill(out, 0);
    CudaDims dim = CudaOneDim(out->size);
    ToSparseByIndicesKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, VecToCuda(strides), indices.ptr, out->size);
}

__global__ void EwisePermuteKernel(const scalar_t* a, scalar_t* out, scalar_t* permutes, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = a[static_cast<int>(permutes[gid])];
    }
}

void EwisePermute(const CudaArray& a, CudaArray* out, const CudaArray& permutes) {
    CudaDims dim = CudaOneDim(out->size);
    EwisePermuteKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, permutes.ptr, out->size);
    cudaFree(permutes.ptr);
}


}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  m.def("sparse_dense_add", SparseDenseAdd);
  m.def("sparse_scalar_add", SparseScalarAdd);
  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_sparse_dense_mul", EwiseSparseDenseMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("ewise_sparse_dense_div", EwiseSparseDenseDiv);
  m.def("scalar_power", ScalarPower);
  
  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);
  
  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);
  m.def("ewise_relu", EwiseRelu);
  m.def("ewise_sigmoid", EwiseSigmoid);

  m.def("matmul", Matmul);
  m.def("sparse_dense_matmul", SparseDenseMatmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("stack", Stack);
  m.def("swap", Swap);
  m.def("to_sparse_by_indices", ToSparseByIndices);

  m.def("ewise_permute", EwisePermute);
}


  // m.def("ewise_sparse_dense_div", EwiseSparseDenseDiv);
  // m.def("sparse_dense_matmul", SparseDenseMatMul); 
