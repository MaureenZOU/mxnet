/* 
 * Written by Xueyan Zou @TuSimple Algorithm Intern
 */

#include "./nonzero-average-inl.h"
#include "./mshadow_op.h"

namespace mshadow{
template<typename Dtype>
inline void nAvgForward(const Tensor<cpu, 4, Dtype> &x,
					   const Tensor<cpu, 4, Dtype> &out,
					   const bool threshold_){
	printf("Not implemented");
}//end SPNForward

template<typename Dtype>
inline void nAvgBackward(const Tensor<cpu, 4, Dtype> &x,
					    const Tensor<cpu, 4, Dtype> &out,
					    const Tensor<cpu, 4, Dtype> &x_diff,
					    const Tensor<cpu, 4, Dtype> &out_diff,
					    const bool threshold_){
	printf("Not implemented");
}//end SPNBackward
}

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(nAvgParam param, int dtype) {
 	Operator* op = NULL;
 	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
 		op = new nAvgOp<cpu, DType>(param);
 	})
	return op;
}

Operator *nAvgProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}


DMLC_REGISTER_PARAMETER(nAvgParam);
MXNET_REGISTER_OP_PROPERTY(nAvg,nAvgProp)
.describe("nonzero average layer")
.add_argument("X", "NDArray-or-Symbol", "X")
.add_arguments(nAvgParam::__FIELDS__());
}
}
