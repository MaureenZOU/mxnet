/* 
 * Written by Xueyan Zou @TuSimple Algorithm Intern
 */

#include "./nonzero-average-inl.h"
#include <assert.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

template <typename Dtype>
__device__ void set_data_na(Dtype * data, int num, int channels, int height, int width, int n, int c, int h, int w, Dtype v)
{
	data[n*channels*height*width + c*height*width + h*width + w] = v;
}

template <typename Dtype>
__device__ Dtype get_data_na(Dtype *data, int num, int channels, int height, int width, int n, int c, int h, int w){
	//spatial-propagation-inl.h:82 -> default configuration of dim is (batch, channel, height, width)
	return data[n*channels*height*width + c*height*width + h*width + w];
}

template <typename Dtype>
__global__ void backward_nonzero_average(const int count, int num, int channels, int height,  int width, const Dtype* X, const Dtype* Avg, Dtype* X_diff, const Dtype* Avg_diff, float threshold_){
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){ 
	//CUDA kernel loop, index trace the current thread	
		int hw_count = height * width;

		int n,c,h,w;
		int temp=index;

		n = temp / hw_count;
		temp = temp % hw_count;
		h = temp / width;
		temp = temp % width;
		w = temp;

		float count = 0.0;
		Dtype zero = 0.0;

		for(c = 0; c < channels; c+=1){
			Dtype x = get_data_na(X,num,channels,height,width,n,c,h,w);
			if(x > threshold_){
				count = count + 1;
			}
		}

		for(c = 0; c < channels; c+=1){
			Dtype x = get_data_na(X,num,channels,height,width,n,c,h,w);
			if(x > threshold_){
				Dtype grad = (1/count)*get_data_na(Avg_diff,num,channels,height,width,n,0,h,w);
				set_data_na(X_diff,num,channels,height,width,n,c,h,w,grad);
				//printf("c[%d] = %lf \n", c, grad);
			}else{
				set_data_na(X_diff,num,channels,height,width,n,c,h,w,zero);
			}
		}
	}
}



template <typename Dtype>
__global__ void forward_nonzero_average(const int count, int num, int channels, int height,  int width, const Dtype* X, Dtype* Avg, float threshold_){
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){ 
	//CUDA kernel loop, index trace the current thread
		int hw_count = height * width;

		int n,c,h,w;
		int temp=index;

		n = temp / hw_count;
		temp = temp % hw_count;
		h = temp / width;
		temp = temp % width;
		w = temp;

		Dtype SUM = 0;
		int count_nonzero = 0;

		for(c = 0; c < channels; c+=1){
			Dtype x = get_data_na(X,num,channels,height,width,n,c,h,w);
			if(x > threshold_){
				SUM = SUM + x;
				count_nonzero +=1;
			}
		}

		Dtype avg =  SUM / count_nonzero;
		set_data_na(Avg,num,channels,height,width,n,0,h,w,avg);
	}
}

template<typename Dtype>
inline void nAvgForward(const Tensor<gpu, 4, Dtype> &data,
					   const Tensor<gpu, 4, Dtype> &out,
					   const float threshold_){

/*get pointer*/	
	const Dtype *X = data.dptr_;
	Dtype *Avg = out.dptr_;
/*END get pointer*/

/*get dimension*/
	//data, g1, g2, g3, out, share the same dimension
	//n_X represent number of X
	const int n_batch = data.size(0);
	const int n_channel = data.size(1);
	const int height = data.size(2);
	const int width = data.size(3);
/*END get dimension*/

/*set cuda system param*/
	const int NUM_THREADS_BLOCK = 512; //CUDA: use 512 threads per block
	const int NUM_BLOCKS_GRID = kMaxGridDim; //CUDA: use largest blocks num per grid
/*END set cuda system param*/
	
	const int n_operation_parallel = height * width * n_batch;
	const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
	const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

	dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
	dim3 dimBlock(NUM_THREADS_BLOCK);

	CheckLaunchParam(dimGrid, dimBlock, "non-zero Average forward"); //check whether dimGrid or dimBlock is out of range
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
	forward_nonzero_average<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, n_batch, n_channel, height, width, X, Avg, threshold_);
}//end SPNForward

template<typename Dtype>
inline void nAvgBackward(const Tensor<gpu, 4, Dtype> &data,
					    const Tensor<gpu, 4, Dtype> &out,
					    const Tensor<gpu, 4, Dtype> &data_diff,
					    const Tensor<gpu, 4, Dtype> &out_diff,
					    const float threshold_){

/*get pointer*/	
	const Dtype *X = data.dptr_;
	const Dtype *Avg = out.dptr_;

	Dtype *X_diff = data_diff.dptr_;
	const Dtype *Avg_diff = out_diff.dptr_;
/*END get pointer*/

/*get dimension*/
	//data, g1, g2, g3, out, share the same dimension
	//n_X represent number of X
	const int n_batch = data.size(0);
	const int n_channel = data.size(1);
	const int height = data.size(2);
	const int width = data.size(3);
/*END get dimension*/

/*set cuda system param*/
	const int NUM_THREADS_BLOCK = 512; //CUDA: use 512 threads per block
	const int NUM_BLOCKS_GRID = kMaxGridDim; //CUDA: use largest blocks num per grid
/*END set cuda system param*/

	const int n_operation_parallel = height * width * n_batch;
	const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
	const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

	dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
	dim3 dimBlock(NUM_THREADS_BLOCK);

	CheckLaunchParam(dimGrid, dimBlock, "non-zero Average backward"); //check whether dimGrid or dimBlock is out of range
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
	backward_nonzero_average<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, n_batch, n_channel, height, width, X, Avg, X_diff, Avg_diff, threshold_);
/*END allocate kernel*/

}//end SPNBackward

} //namespace cuda
template<typename Dtype>
inline void nAvgForward(const Tensor<gpu, 4, Dtype> &data,
					   const Tensor<gpu, 4, Dtype> &out,
					   const float threshold_){
	cuda::nAvgForward(data, out, threshold_);
}

template<typename Dtype>
inline void nAvgBackward(const Tensor<gpu, 4, Dtype> &data,
					    const Tensor<gpu, 4, Dtype> &out,
					    const Tensor<gpu, 4, Dtype> &data_diff,
					    const Tensor<gpu, 4, Dtype> &out_diff,
					    const float threshold_){
	cuda::nAvgBackward(data, out, data_diff, out_diff, threshold_);
}
} //namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(nAvgParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new nAvgOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
