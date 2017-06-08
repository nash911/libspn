#ifndef TENSORFLOW_USEROPS_SCATTER_VALUES_FUNCTOR_H_
#define TENSORFLOW_USEROPS_SCATTER_VALUES_FUNCTOR_H_

#include <unordered_set>
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

using namespace std;

namespace tensorflow
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor
{
//--Helper method to count and copy using memcpy()--//
template <typename T, typename IndT>
Status PadAndCopy(const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output)
{
  const int64 indices_rows = indices.dimension(0);
  const int64 indices_cols = indices.dimension(1);
  const int64 params_rows = params.dimension(0);
  const int64 params_cols = params.dimension(1);

  // TODO: May be this is not needed, check it.
  /*unordered_set<IndT> unique_ind(&indices(0), &indices(indices_size));
  if (unique_ind.size() != indices_size)
  {
    return errors::InvalidArgument(
        "Indices cannot contain duplicates.", " Total no. of indices: ",
        indices_size, " != no. of unique indices: ", unique_ind.size(), ".");
  }*/

  //--Declare padding element as a double, and set it to default value 0.0--//
  double pad_elem = 0.0;

  //--Vector containing padding elements. Size of this vector = num_out_cols--//
  gtl::InlinedVector<T, 4> pad_elem_vec(num_out_cols, (T)pad_elem);


  std::vector<IndT> out_indices(
      num_out_cols, -1);  //--Here '-1' refers to padding column(s)--//

  //--Debugging flag disabled by default--//
  #if EXEC_TIME_CALC
    clock_t start, end;
    float time_taken;
    start = clock();
  #endif  // EXEC_TIME_CALC
  for (int row = 0; row < indices_rows; row++)
  {
    for (int col = 0; col < indices_cols; col++)
    {
      //--Check indices[r][c] ∈ (0, num_out_cols]--//
      if (!FastBoundsCheck(indices(row, col), num_out_cols))
      {
        return errors::InvalidArgument("Indices(", row, ", ", col, "): ", indices(row, col),
                                       " is not in range (0, ", num_out_cols,
                                       "].");
      }

      /*
      //--Mem-copy an entire column of padding elements--//
      memcpy(&output(row, col, 0), &pad_elem_vec[0], (num_out_cols * sizeof(T)));

      //--Mem-copy a single element from params tensor--//
      memcpy(&output(row, col, indices(row, col)), &params(row, col), sizeof(T));*/

      //--Mem-copy an entire column of padding elements--//
      memcpy(&output(((row * indices_cols) + col), 0), &pad_elem_vec[0], (num_out_cols * sizeof(T)));

      //--Mem-copy a single element from params tensor--//
      memcpy(&output(((row * indices_cols) + col), indices(row, col)), &params(row, col), sizeof(T));
    }
  }

  //--MemCpy padding element '0' to output tensor--//
  //double pad_elem = 0.0;
  //memset(&output(0, 0), 0, ((indices_rows * indices_cols * num_out_cols) * sizeof(T)));

  //--Debugging flag disabled by default--//
  #if EXEC_TIME_CALC
    end = clock();
    time_taken =
        (((float)(end - start)) / CLOCKS_PER_SEC) * 1000.0;  //--Milliseconds//
    std::cout << "CPU - Time Taken: " << time_taken << " ms" << endl;
  #endif  // EXEC_TIME_CALC

  return Status::OK();
}

template <typename T, typename IndT>
struct ScatterValuesFunctorCPU
{
  Status operator()(const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output)
  {
    return PadAndCopy<T, IndT>(params, indices,
                                 num_out_cols, output);
  }
};

template <typename Device, typename T, typename IndT>
struct ScatterValuesFunctor
{
  Status operator()(const Device& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output);
};

template <typename T, typename IndT>
struct ScatterValuesFunctor<CPUDevice, T, IndT>
{
  Status operator()(const CPUDevice& dvc,
                    const typename TTypes<T>::ConstMatrix& params,
                    const typename TTypes<IndT>::ConstMatrix& indices,
                    const IndT& num_out_cols,
                    typename TTypes<T>::Matrix& output)
  {
    return ScatterValuesFunctorCPU<T, IndT>()(params, indices,
                                               num_out_cols, output);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_SCATTER_VALUES_FUNCTOR_H_
