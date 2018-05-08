/* 
 * Written by Xueyan Zou @TuSimple Algorithm Intern
 */

#ifndef MXNET_OPERATOR_SCN_INL_H
#define MXNET_OPERATOR_SCN_INL_H

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
//??unimplemented, please refer roi_pooling and conv later

namespace mxnet {
namespace op {

namespace scn { //default: three way connection
enum scnInputs {kData, kG1, kG2, kG3, kC}; //kG1, kG2, kG3 is the gi maxtrix for h(t) = (1-sum(g_i(t))) * x_i(t) + sum(g_i(t) * h_i(t-1))
enum scnOutputs {kOut};
}

struct ScnParam : public dmlc::Parameter<ScnParam> {
	bool horizontal;
	bool reverse;

	DMLC_DECLARE_PARAMETER(ScnParam) {
    DMLC_DECLARE_FIELD(horizontal).set_default(false)
    .describe("Identify the direction of propagation");
	DMLC_DECLARE_FIELD(reverse).set_default(false)
    .describe("Identify the direction of propagation");
	}

	/*
	 * horizontal = false, reverse = false top -> bottom
	 * horizontal = false, reverse = true bottom -> top
	 * horizontal = true, reverse = False left -> right
	 * horizontal = true, reverse = true right -> left
	 */
};


template<typename xpu, typename DType>
class ScnOp : public Operator {
public:
	explicit ScnOp(ScnParam param) {
		param_ = param;
  	}
	
	virtual void Forward(const OpContext &ctx, //use for register stream
						 const std::vector<TBlob> &in_data, //input data
						 const std::vector<OpReqType> &req, //TBA below
						 const std::vector<TBlob> &out_data, //output data
						 const std::vector<TBlob> &aux_states){ //unused most of the time
		using namespace mshadow;
		using namespace mshadow::expr;

	//req store the flag how to act on the value after the operation
	// [null] -> skip calculating the corresponding output tensor
	// [write] -> overwriting the values in the output tensor
	// [add] -> adding the calculated value
	// req[scn:kData] stands for the corresponding operation on the data	

	/*check dimension*/
		// [] stands for matrix
		size_t expected_in = 5; //in_data includes [g1], [g2], [g3], [x]
		size_t expected_out = 1; //out_data includes [h]

		//check number of inputs and outputs
		CHECK_EQ(in_data.size(), expected_in); //check correctness of input size
		CHECK_EQ(out_data.size(), expected_out); //check correctness of output size

		//check shape: [g1] = [g2] = [g3] = [x] = [h]
		CHECK_EQ(in_data[scn::kData].shape_[0], in_data[scn::kG1].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], in_data[scn::kG2].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], in_data[scn::kG3].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], in_data[scn::kC].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], out_data[scn::kOut].shape_[0]);
	/*END check dimension*/

	/*register stream*/
		Stream<xpu> *s = ctx.get_stream<xpu>();
	/*END register stream*/

	/*load parameter*/		
		//Tensor<xpu, dim, DType>: the default configuration of dim is (batch, channel, height, width)
		Tensor<xpu, 4, DType> data = in_data[scn::kData].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g1 = in_data[scn::kG1].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g2 = in_data[scn::kG2].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g3 = in_data[scn::kG3].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> c = in_data[scn::kC].get<xpu, 4, DType>(s);

		Tensor<xpu, 4, DType> out = out_data[scn::kOut].get<xpu, 4, DType>(s);
	/*END load parameter*/


	/*check memory continuous*/
		CHECK_EQ(data.CheckContiguous(), true);
		CHECK_EQ(g1.CheckContiguous(), true);
		CHECK_EQ(g2.CheckContiguous(), true);
		CHECK_EQ(g3.CheckContiguous(), true);
		CHECK_EQ(c.CheckContiguous(), true);
		CHECK_EQ(out.CheckContiguous(), true);
	/*END check memory continuous*/


	/*operation*/
		SCNForward(data, g1, g2, g3, c, out, param_.horizontal, param_.reverse);
	/*END operation*/

	}//end foward

	virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_states){
		using namespace mshadow;
		using namespace mshadow::expr;

	/*check dimension*/
		// [] stands for matrix
		size_t expected_in = 5; //in_data includes [g1], [g2], [g3], [x]
		size_t expected_out = 1; //out_data includes [h]

		//check number of inputs and outputs
		CHECK_EQ(in_data.size(), expected_in); //check correctness of input size
		CHECK_EQ(out_data.size(), expected_out); //check correctness of output size

		//check shape: [g1_diff] = [g2_diff] = [g3_diff] = [x_diff] = [h_diff]
		CHECK_EQ(in_data[scn::kData].shape_[0], in_grad[scn::kG1].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], in_grad[scn::kG2].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], in_grad[scn::kG3].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], in_grad[scn::kData].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], in_grad[scn::kC].shape_[0]);
		CHECK_EQ(in_data[scn::kData].shape_[0], out_grad[scn::kOut].shape_[0]);
	/*END check dimension*/
	
	/*register stream*/
		Stream<xpu> *s = ctx.get_stream<xpu>();
	/*END register stream*/

	/*load parameter*/		
		//Tensor<xpu, dim, DType>: the default configuration of dim is (batch, channel, height, width)
		Tensor<xpu, 4, DType> data = in_data[scn::kData].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g1 = in_data[scn::kG1].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g2 = in_data[scn::kG2].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g3 = in_data[scn::kG3].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> c = in_data[scn::kC].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> out = out_data[scn::kOut].get<xpu, 4, DType>(s);

		Tensor<xpu, 4, DType> data_diff = in_grad[scn::kData].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g1_diff = in_grad[scn::kG1].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g2_diff = in_grad[scn::kG2].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> g3_diff = in_grad[scn::kG3].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> c_diff = in_grad[scn::kC].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> out_diff = out_grad[scn::kOut].get<xpu, 4, DType>(s);
	/*END load parameter*/

	/*operation*/
		SCNBackward(data, g1, g2, g3, c, out, data_diff, g1_diff, g2_diff, g3_diff, c_diff, out_diff, param_.horizontal, param_.reverse);
	/*END operation*/

	}

private:
	ScnParam param_;
}; //class ScnOp

template<typename xpu>
Operator* CreateOp(ScnParam param, int dtype);


#if DMLC_USE_CXX11
class ScnProp : public OperatorProperty {

public:
	std::vector<std::string> ListArguments() const override {
    	return {"X", "G1", "G2", "G3", "C"};
  	}

	std::vector<std::string> ListOutputs() const override {
		return {"H"};
  	}

	int NumOutputs() const override {
		return 1; //H
	}

	int NumVisibleOutputs() const override {
		return 1; //H
	}

	void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
		param_.Init(kwargs);
	}

	bool InferShape(std::vector<TShape> *in_shape,
				    std::vector<TShape> *out_shape,
				    std::vector<TShape> *aux_shape) const override {
		using namespace mshadow;
		CHECK_EQ(in_shape->size(), 5U) << "Input:[X, G1, G2, G3, C]";

    	const TShape &dshape = in_shape->at(scn::kData);
    	out_shape->clear();
    	out_shape->push_back(dshape);

    	return true;
    }	

	std::map<std::string, std::string> GetParams() const override {
		return param_.__DICT__();
	}

	OperatorProperty* Copy() const override {
		auto ptr = new ScnProp();
		ptr->param_ = param_;
		return ptr;
	}

	std::string TypeString() const override {
		return "SCN";
	}

	// decalre dependency and inplace optimization options
	std::vector<int> DeclareBackwardDependency(
		const std::vector<int> &out_grad,
		const std::vector<int> &in_data,
		const std::vector<int> &out_data) const override {
		return {out_grad[scn::kOut], in_data[scn::kData], in_data[scn::kG1], in_data[scn::kG2], in_data[scn::kG3], in_data[scn::kC]};
	}

	std::vector<std::pair<int, void*> > BackwardInplaceOption(
		const std::vector<int> &out_grad,
		const std::vector<int> &in_data,
		const std::vector<int> &out_data,
		const std::vector<void*> &in_grad) const override {
		return {{in_data[scn::kC], in_grad[scn::kData]}, {in_data[scn::kC], in_grad[scn::kG1]}, {in_data[scn::kC], in_grad[scn::kG2]}, {in_data[scn::kC], in_grad[scn::kG3]}, {in_data[scn::kOut], in_grad[scn::kG1]}, {in_data[scn::kOut], in_grad[scn::kG2]}, {in_data[scn::kOut], in_grad[scn::kG3]}, {in_data[scn::kG1], in_grad[scn::kC]}, {in_data[scn::kG2], in_grad[scn::kC]}, {in_data[scn::kG3], in_grad[scn::kC]}, {in_data[scn::kOut], in_grad[scn::kC]}};
	}

 
	Operator* CreateOperator(Context ctx) const override {
		LOG(FATAL) << "Not Implemented.";
		return NULL;
	}

	Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

private:
	ScnParam param_;
};
#endif //DMLC_USE_CXX11
} //namespace op
} //namespace mxnet
#endif //MXNET_OPERATOR_SCN_INL_H
