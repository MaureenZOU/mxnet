/* 
 * Written by Xueyan Zou @TuSimple Algorithm Intern
 */

#ifndef MXNET_OPERATOR_nAvg_INL_H
#define MXNET_OPERATOR_nAvg_INL_H

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

namespace mxnet {
namespace op {

namespace nAvg {
enum nAvgInputs {kX}; 
enum nAvgOutputs {kOut};
}

struct nAvgParam : public dmlc::Parameter<nAvgParam> {
	float threshold;

	DMLC_DECLARE_PARAMETER(nAvgParam) {
    DMLC_DECLARE_FIELD(threshold).set_default(1.0)
    .describe("Threshold to omit calculation");
	}
};


template<typename xpu, typename DType>
class nAvgOp : public Operator {
public:
	explicit nAvgOp(nAvgParam param) {
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
	// req[spn:kData] stands for the corresponding operation on the data	

	/*check dimension*/
		// [] stands for matrix
		size_t expected_in = 1; //in_data includes [g1], [g2], [g3], [x]
		size_t expected_out = 1; //out_data includes [h]

		//check number of inputs and outputs
		CHECK_EQ(in_data.size(), expected_in); //check correctness of input size
		CHECK_EQ(out_data.size(), expected_out); //check correctness of output size

		CHECK_EQ(out_data[nAvg::kOut].shape_[2], in_data[nAvg::kX].shape_[2]);
	/*END check dimension*/

	/*register stream*/
		Stream<xpu> *s = ctx.get_stream<xpu>();
	/*END register stream*/

	/*load parameter*/		
		//Tensor<xpu, dim, DType>: the default configuration of dim is (batch, channel, height, width)
		Tensor<xpu, 4, DType> x = in_data[nAvg::kX].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> out = out_data[nAvg::kOut].get<xpu, 4, DType>(s);
	/*END load parameter*/


	/*check memory continuous*/
		CHECK_EQ(out.CheckContiguous(), true);
		CHECK_EQ(x.CheckContiguous(), true);
	/*END check memory continuous*/


	/*operation*/
		nAvgForward(x, out, param_.threshold);
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
		size_t expected_in = 1; //in_data includes [g1], [g2], [g3], [x]
		size_t expected_out = 1; //out_data includes [h]

		//check number of inputs and outputs
		CHECK_EQ(in_data.size(), expected_in); //check correctness of input size
		CHECK_EQ(out_data.size(), expected_out); //check correctness of output size

		//check shape: [g1_diff] = [g2_diff] = [g3_diff] = [x_diff] = [h_diff]
		CHECK_EQ(out_data[nAvg::kOut].shape_[0], in_grad[nAvg::kX].shape_[0]);
		CHECK_EQ(out_data[nAvg::kOut].shape_[0], out_grad[nAvg::kOut].shape_[0]);
	/*END check dimension*/
	
	/*register stream*/
		Stream<xpu> *s = ctx.get_stream<xpu>();
	/*END register stream*/

	/*load parameter*/		
		//Tensor<xpu, dim, DType>: the default configuration of dim is (batch, channel, height, width)
		Tensor<xpu, 4, DType> x = in_data[nAvg::kX].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> out = out_data[nAvg::kOut].get<xpu, 4, DType>(s);

		Tensor<xpu, 4, DType> x_diff = in_grad[nAvg::kX].get<xpu, 4, DType>(s);
		Tensor<xpu, 4, DType> out_diff = out_grad[nAvg::kOut].get<xpu, 4, DType>(s);
	/*END load parameter*/

	/*operation*/
		nAvgBackward(x, out, x_diff, out_diff, param_.threshold);
	/*END operation*/

	}

private:
	nAvgParam param_;
}; //class SpnOp

template<typename xpu>
Operator* CreateOp(nAvgParam param, int dtype);


#if DMLC_USE_CXX11
class nAvgProp : public OperatorProperty {

public:
	std::vector<std::string> ListArguments() const override {
    	return {"X"};
  	}

	std::vector<std::string> ListOutputs() const override {
		return {"Out"};
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
		CHECK_EQ(in_shape->size(), 1U) << "Input:[X]";

    	const TShape &dshape = in_shape->at(nAvg::kX);
    	out_shape->clear();
    	out_shape->push_back(dshape);

    	return true;
    }	

	std::map<std::string, std::string> GetParams() const override {
		return param_.__DICT__();
	}

	OperatorProperty* Copy() const override {
		auto ptr = new nAvgProp();
		ptr->param_ = param_;
		return ptr;
	}

	std::string TypeString() const override {
		return "nAvg";
	}

	Operator* CreateOperator(Context ctx) const override {
		LOG(FATAL) << "Not Implemented.";
		return NULL;
	}

	Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

private:
	nAvgParam param_;
};
#endif //DMLC_USE_CXX11
} //namespace op
} //namespace mxnet
#endif //MXNET_OPERATOR_SPN_INL_H
