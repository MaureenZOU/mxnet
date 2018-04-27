/* 
 * Written by Xueyan Zou @TuSimple Algorithm Intern
 */

#include "./spatial-propagation-inl.h"
#include "./mshadow_op.h"

namespace mshadow { 
template<typename Dtype>
inline void SPNForward(const Tensor<gpu, 4, Dtype> &data,
					   const Tensor<gpu, 4, Dtype> &g1,
					   const Tensor<gpu, 4, Dtype> &g2,
					   const Tensor<gpu, 4, Dtype> &g3,
					   const Tensor<gpu, 4, Dtype> &out,
					   const bool horizontal_,
					   const bool reverse_){
	printf("Not implemented");
}//end SPNForward

template<typename Dtype>
inline void SPNBackward(const Tensor<gpu, 4, Dtype> &data,
					    const Tensor<gpu, 4, Dtype> &g1,
					    const Tensor<gpu, 4, Dtype> &g2,
					    const Tensor<gpu, 4, Dtype> &g3,
					    const Tensor<gpu, 4, Dtype> &out,
					    const Tensor<gpu, 4, Dtype> &data_diff,
					    const Tensor<gpu, 4, Dtype> &g1_diff,
					    const Tensor<gpu, 4, Dtype> &g2_diff,
					    const Tensor<gpu, 4, Dtype> &g3_diff,
					    const Tensor<gpu, 4, Dtype> &out_diff,
					    const bool horizontal_,
					    const bool reverse_){
	printf("Not implemented");
}//end SPNBackward

}  // namespace mshadow
namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SpnParam param, int dtype) {
  return new SpnOp<cpu, DType>(param);
}