#include "./spatial-propagation-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(SpnParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
	LOG(FATAL) << "CPU unimplemented";
}
}
}