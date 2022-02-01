#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

void backtrack(std::vector<std::vector<int32_t>>& res, std::vector<int32_t>& cur, std::vector<int32_t>& shape) {
    if (cur.size() == shape.size()) {
        res.push_back(cur);
    } else {
        int index = cur.size();
        int32_t s = shape[index];
        for (int32_t i = 0; i < s; i++) {
            cur.push_back(i);
            backtrack(res, cur, shape);
            cur.pop_back();
        }
    }
}

std::vector<std::vector<int32_t>> all_permutations(std::vector<int32_t>& shape) {
    std::vector<std::vector<int32_t>> res;
    std::vector<int32_t> cur;
    backtrack(res, cur, shape);
    return res;
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
    size_t cnt = 0;
    std::vector<std::vector<int32_t>> permutations = all_permutations(shape);
    for (std::vector<int32_t> pos : permutations) {
        int32_t _offset = 0;
        for (int idx = 0; idx < pos.size(); idx++) {
            _offset += pos[idx] * strides[idx];
        }
        out->ptr[cnt++] = a.ptr[_offset + offset];
    }
    out->size = cnt;
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
    size_t cnt = 0;
    std::vector<std::vector<int32_t>> permutations = all_permutations(shape);
    for (std::vector<int32_t> pos : permutations) {
        int32_t _offset = 0;
        for (int idx = 0; idx < pos.size(); idx++) {
            _offset += pos[idx] * strides[idx];
        }
        out->ptr[_offset + offset] = a.ptr[cnt++];
    }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
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
    size_t cnt = 0;
    std::vector<std::vector<int32_t>> permutations = all_permutations(shape);
    for (std::vector<int32_t> pos : permutations) {
        int32_t _offset = 0;
        for (int idx = 0; idx < pos.size(); idx++) {
            _offset += pos[idx] * strides[idx];
        }
        out->ptr[_offset + offset] = val;
    }
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
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

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = pow(a.ptr[i], val);
  }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    if (a.ptr[i] == 0.0 && b.ptr[i] == 0.0) {
        out->ptr[i] = 0.0;
    } else {
        out->ptr[i] = a.ptr[i] / b.ptr[i];
    }
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = tanh(a.ptr[i]);
  }
}

void EwiseRelu(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    if (a.ptr[i] >= 0.0) out->ptr[i] = a.ptr[i];
    else out->ptr[i] = 0.0; 
  }
}

void EwiseSigmoid(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = 1.0 / (1.0 + exp(-a.ptr[i]));
  }
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == b.ptr[i];
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] == val;
  }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= b.ptr[i];
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] >= val;
  }
}

void EwiseNe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] != b.ptr[i];
  }
}

void ScalarNe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] != val;
  }
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, int32_t m, int32_t n,
            int32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: coolumns of b / out
   */

    for (int32_t mm = 0; mm < m; mm++) {
        for (int32_t pp = 0; pp < p; pp++) {
            scalar_t sum = 0.0;
            for (int32_t nn = 0; nn < n; nn++) {
                sum += a.ptr[mm * n + nn] * b.ptr[nn * p + pp];
            }
            out->ptr[mm * p + pp] = sum;
        }
    }
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array siwll be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

    a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
    b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
    out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);
    for (int32_t i = 0; i < TILE; i++) {
        for (int32_t j = 0; j < TILE; j++) {
            scalar_t sum = 0.0;
            for (int32_t k = 0; k < TILE; k++) {
                sum += a[i * TILE + k] * b[k * TILE + j];
            }
            out[i * TILE + j] += sum;
        }
    }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, int32_t m,
                 int32_t n, int32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: coolumns of b / out
   *
   */
    memset(out->ptr, 0, m * p * ELEM_SIZE);
    // for (int32_t os = 0; os < m*p; os++) {
    //     out->ptr[os] = 0.0;
    // }
    for (int32_t i = 0; i < m/TILE; i++) {
        for (int32_t j = 0; j < p/TILE; j++) {
            scalar_t* _out = out->ptr + i * p * TILE + j * TILE * TILE;
            for (int32_t k = 0; k < n/TILE; k++) {
                scalar_t* _a = a.ptr + i * n * TILE + k * TILE * TILE;
                scalar_t* _b = b.ptr + k * p * TILE + j * TILE * TILE;
                AlignedDot(_a, _b, _out);
            }
        }
    }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

    size_t offset = 0;
    size_t out_offset = 0;
    while (offset < a.size) {
        scalar_t cur_max = a.ptr[offset];
        for (int32_t i = 0; i < reduce_size; i++) {
            scalar_t cur = a.ptr[offset++];
            if (cur > cur_max) {
                cur_max = cur;
            }
        }
        out->ptr[out_offset++] = cur_max;
    }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

    for (size_t i = 0; i < out->size; i++) {
        out->ptr[i] = 0.0;
        for (size_t j = i * reduce_size; j < (i + 1) * reduce_size; j++) {
            out->ptr[i] += a.ptr[j];
        }
    }
}

void Stack(const std::vector<AlignedArray*>& arrays, AlignedArray* out, std::vector<int32_t> shape, std::vector<int32_t> strides, std::vector<int32_t> new_shape, int32_t axis) {
    size_t cnt = 0;
    std::vector<std::vector<int32_t>> permutations = all_permutations(new_shape);
    for (std::vector<int32_t> pos : permutations) {
        int32_t _offset = 0;
        int old_shape_idx = 0;
        for (int32_t idx = 0; idx < pos.size(); idx++) {
            if (idx == axis) continue;
            _offset += pos[idx] * strides[old_shape_idx];
            ++old_shape_idx;
        }
        out->ptr[cnt++] = arrays[pos[axis]]->ptr[_offset];
    }
}

void SparseScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out, std::vector<int32_t> strides, const AlignedArray& indices) {
    size_t size = out->size;
    for (size_t i = 0; i < size; i++) {
        out->ptr[i] = val;
    }
    size_t ele_cnt = indices.size / 2;
    for (size_t i = 0; i < ele_cnt; i++) {
        size_t offset = indices.ptr[i] * strides[0] + indices.ptr[i+ele_cnt] * strides[1];
        out->ptr[offset] += a.ptr[i];
    }
}

void SparseDenseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, std::vector<int32_t> strides, const AlignedArray& indices) {
    size_t size = out->size;
    for (size_t i = 0; i < size; i++) {
        out->ptr[i] = b.ptr[i];
    }
    size_t ele_cnt = indices.size / 2;
    for (size_t i = 0; i < ele_cnt; i++) {
        size_t offset = indices.ptr[i] * strides[0] + indices.ptr[i+ele_cnt] * strides[1];
        out->ptr[offset] += a.ptr[i];
    }
}

void EwiseSparseDenseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, std::vector<int32_t> strides, const AlignedArray& indices) {
    size_t size = out->size;
    Fill(out, 0);
    size_t ele_cnt = indices.size / 2;
    for (size_t i = 0; i < ele_cnt; i++) {
        size_t offset = indices.ptr[i] * strides[0] + indices.ptr[i+ele_cnt] * strides[1];
        out->ptr[offset] += a.ptr[i] * b.ptr[offset]; 
    }
}

void EwiseSparseDenseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, std::vector<int32_t> strides, const AlignedArray& indices) {
    size_t size = out->size;
    Fill(out, 0);
    size_t ele_cnt = indices.size / 2;
    for (size_t i = 0; i < ele_cnt; i++) {
        size_t offset = indices.ptr[i] * strides[0] + indices.ptr[i+ele_cnt] * strides[1];
        out->ptr[offset] += a.ptr[i] / b.ptr[offset]; 
    }
}

void SparseDenseMatMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, const AlignedArray& indices, int32_t m, int32_t n,
            int32_t p) {
    size_t size = out->size;
    Fill(out, 0);
    size_t ele_cnt = indices.size / 2;
    for(size_t i = 0; i < ele_cnt; i++) {
        size_t idx_x = indices.ptr[i];
        size_t idx_y = indices.ptr[i+ele_cnt];
        for(size_t k = 0; k < p; k++) {
            size_t offset = idx_x * p + k;
            out->ptr[offset] += a.ptr[i]  * b.ptr[idx_y * p + k];
        }
    }
}

void Swap(const AlignedArray& a, AlignedArray* out) {
    size_t size = out->size;
    for (size_t i = 0; i < size / 2; i++) {
        out->ptr[i] = a.ptr[i+size/2];
    }
    for (size_t j = size / 2; j < size; j++) {
        out->ptr[j] = a.ptr[j-size/2];
    }
}

void ToSparseByIndices(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> strides, const AlignedArray& indices) {
    size_t ele_cnt = indices.size / 2;
    Fill(out, 0);
    for (size_t i = 0; i < ele_cnt; i++) {
        size_t offset = indices.ptr[i] * strides[0] + indices.ptr[i+ele_cnt] * strides[1];
        out->ptr[i] = a.ptr[offset];
    }
}


}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);
  
  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);
  m.def("ewise_ne", EwiseNe);
  m.def("scalar_ne", ScalarNe);
  
  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);
  m.def("ewise_relu", EwiseRelu);
  m.def("ewise_sigmoid", EwiseSigmoid);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("stack", Stack);

  m.def("sparse_scalar_add", SparseScalarAdd);
  m.def("sparse_dense_add", SparseDenseAdd);
  m.def("ewise_sparse_dense_mul", EwiseSparseDenseMul);
  m.def("ewise_sparse_dense_div", EwiseSparseDenseDiv);
  m.def("sparse_dense_matmul", SparseDenseMatMul); 
  m.def("swap", Swap); 

  m.def("to_sparse_by_indices", ToSparseByIndices);
}
