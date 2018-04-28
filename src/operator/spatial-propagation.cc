/* 
 * Written by Xueyan Zou @TuSimple Algorithm Intern
 */

#include "./spatial-propagation-inl.h"
#include "./mshadow_op.h"

namespace mshadow{
template<typename Dtype>
inline void SPNForward(const Tensor<cpu, 4, Dtype> &data,
					   const Tensor<cpu, 4, Dtype> &g1,
					   const Tensor<cpu, 4, Dtype> &g2,
					   const Tensor<cpu, 4, Dtype> &g3,
					   const Tensor<cpu, 4, Dtype> &out,
					   const bool horizontal_,
					   const bool reverse_){
	printf("Not implemented");
}//end SPNForward

template<typename Dtype>
inline void SPNBackward(const Tensor<cpu, 4, Dtype> &data,
					    const Tensor<cpu, 4, Dtype> &g1,
					    const Tensor<cpu, 4, Dtype> &g2,
					    const Tensor<cpu, 4, Dtype> &g3,
					    const Tensor<cpu, 4, Dtype> &out,
					    const Tensor<cpu, 4, Dtype> &data_diff,
					    const Tensor<cpu, 4, Dtype> &g1_diff,
					    const Tensor<cpu, 4, Dtype> &g2_diff,
					    const Tensor<cpu, 4, Dtype> &g3_diff,
					    const Tensor<cpu, 4, Dtype> &out_diff,
					    const bool horizontal_,
					    const bool reverse_){
	printf("Not implemented");
}//end SPNBackward
}

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SpnParam param, int dtype) {
 	Operator* op = NULL;
 	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
 		op = new SpnOp<cpu, DType>(param);
 	})
	return op;
}

Operator *SpnProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}


DMLC_REGISTER_PARAMETER(SpnParam);
MXNET_REGISTER_OP_PROPERTY(SPN,SpnProp)
.describe("Spatial propagation Network")
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_argument("g1", "NDArray-or-Symbol", "G1")
.add_argument("g2", "NDArray-or-Symbol", "G2")
.add_argument("g3", "NDArray-or-Symbol", "G3")
.add_arguments(SpnParam::__FIELDS__());
}
}
