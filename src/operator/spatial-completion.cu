/* 
 * Written by Xueyan Zou @TuSimple Algorithm Intern
 */

#include "./spatial-completion-inl.h"
#include <assert.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

__device__ void get_gate_idx_xy(int h1, int w1, int h2, int w2, int* out, bool horizontal, bool reverse)
{
	if(horizontal && ! reverse) // left -> right
	{
		if(w1>w2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(horizontal && reverse)  // right -> left
	{
		if(w1<w2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(!horizontal && !reverse)  // top  -> bottom
	{
		if(h1>h2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}
	if(!horizontal && reverse)  // bottom -> top
	{
		if(h1<h2)
		{
			out[0]=h1;
			out[1]=w1;
		}
		else
		{
			out[0]=h2;
			out[1]=w2;
		}
	}

}

template <typename Dtype>
__device__ void set_gate_xy(Dtype* data, int num, int channels, int height, int width, int n, int c, int h1, int w1, int h2, int w2, Dtype v, bool horizontal, bool reverse)
{
	if(h1<0 || h1 >=height) //redundant
		return ; //redundant
	if(w1<0 || w1 >= width) //redundant
		return ; //redundant
	if(h2<0 || h2 >=height) //redundant
		return ; //redundant
	if(w2<0 || w2 >= width) //redundant
		return ; //redundant

	int idx[2];
	get_gate_idx_xy(h1, w1, h2, w2, idx, horizontal, reverse);
	
	int h = idx[0];
	int w = idx[1];

	data[n*channels*height*width + c*height*width + h*width + w] = v;
}

template <typename Dtype> //this function is modified by xueyan
__device__ Dtype get_gate_xy(Dtype * data, int num, int channels, int height, int width, int n, int c, int h1, int w1, int h2, int w2, bool horizontal, bool reverse){
	//handle index out of range
	if(h1<0 || h1 >=height) //redundant
		return 0; //redundant
	if(w1<0 || w1 >= width) //redundant
		return 0; //redundant
	if(h2<0 || h2 >=height)
		return 0;
	if(w2<0 || w2 >= width)
		return 0;

	int idx[2];
	get_gate_idx_xy(h1, w1, h2, w2, idx, horizontal, reverse);
	
	int h = idx[0];
	int w = idx[1];

	return data[n*channels*height*width + c*height*width + h*width + w];
}


template <typename Dtype>
__device__ void set_data_xy(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w, Dtype v)
{
	//modify by xueyan, assert error.
	if(h<0 || h >=height)
		assert(0);
	if(w<0 || w >= width)
		assert(0);

	data[n*channels*height*width + c*height*width + h*width + w] = v;
}

template <typename Dtype>
__device__ Dtype get_data_xy(Dtype *data, int num, int channels, int height, int width, int n, int c, int h, int w){
	//handle index out of range
	if(h<0 || h >=height)
		return 0;
	if(w<0 || w >= width)
		return 0;
	
	//spatial-propagation-inl.h:82 -> default configuration of dim is (batch, channel, height, width)
	return data[n*channels*height*width + c*height*width + h*width + w];
}

/*LEFT->RIGHT*/
/*h(t) = (1-sum(g_i(t))) * x_i(t) + sum(g_i(t) * h_i(t-1))*/
template <typename Dtype>
__global__ void forward_one_col_left_right(const int count, int T, int num, int channels, int height, int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, Dtype* H, bool horizontal, bool reverse){
//count -> total number of threads; T -> current_row/current_column; num -> total num batch
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){
		//CUDA kernel loop, index trace the current thread
		int hc_count = height * channels; //in order to calculate batch size 

		int n,c,h,w; //w->current_col; n->current_batch; c->current_channel;
		int temp = index;
		w = T;
		n = temp / hc_count;
		temp = temp % hc_count;
		c = temp / height;
		temp = temp % height;
		h = temp;
		//locate the pixel as (n,c,h,w);

		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w); //x
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c

	
		//modify logic by xueyan
		Dtype g_data_1 = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse); //g_1(t)
		Dtype h_minus1_data_1 = get_data_xy(H,num,channels,height,width,n,c,h-1,w-1); //h_1(t-1)
		Dtype h1_minus1 = g_data_1 * h_minus1_data_1; //g_1(t)*h_1(t-1)

		Dtype g_data_2 = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse); //g_2(t)
		Dtype h_minus1_data_2 = get_data_xy(H,num,channels,height,width,n,c,h,w-1); //h_2(t-1)
		Dtype h2_minus1 = g_data_2 * h_minus1_data_2; //g_2(t)*h_2(t-1)

		Dtype g_data_3 = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse); //g_3(t)
		Dtype h_minus1_data_3 = get_data_xy(H,num,channels,height,width,n,c,h+1,w-1); //h_3(t-1)
		Dtype h3_minus1 = g_data_3 * h_minus1_data_3; //g_3(t)*h_3(t-1)

		Dtype h_hype = (h1_minus1 + h2_minus1 + h3_minus1) * (1 - c_data); //sum(g_i(t)*h_i(t-1)) * (1-c) = (g_1(t)*h_1(t-1)+g_2(t)*h_2(t-1)+g_3(t)*h_3(t-1))*(1-c) 
		Dtype x_hype = c_data * x_data; //c * x

		Dtype h_data = x_hype + h_hype; //c * x_i(t) + sum(g_i(t) * h_i(t-1))*(1-c)

		set_data_xy(H,num,channels,height,width,n,c,h,w,h_data); //set H data at point x
	}
}
/*END h(t) = (1-sum(g_i(t))) * x_i(t) + sum(g_i(t) * h_i(t+1))*/

/*RIGHT->LEFT*/
/*h(t) = c * x_i(t) + (1-c) * sum(g_i(t) * h_i(t+1))*/
template <typename Dtype>
__global__ void forward_one_col_right_left(const int count, int T, int num, int channels, int height, int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, Dtype* H, bool horizontal, bool reverse){
//count -> total number of threads; T -> current_row/current_column; num -> total num batch
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){
		//CUDA kernel loop, index trace the current thread
		int hc_count = height * channels;

		int n,c,h,w; //w->current_col; n->current_batch; c->current_channel;
		int temp = index;
		w = T;
		n = temp / hc_count;
		temp = temp % hc_count;
		c = temp / height;
		temp = temp % height;
		h = temp;
		//locate the pixel as (n,c,h,w);
	
		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w); //x
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c

		//modify logic by xueyan
		Dtype g_data_1 = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse); //g_1(t)
		Dtype h_minus1_data_1 = get_data_xy(H,num,channels,height,width,n,c,h-1,w+1); //h_1(t+1)
		Dtype h1_minus1 = g_data_1 * h_minus1_data_1; //g_1(t)*h_1(t+1)

		Dtype g_data_2 = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse); //g_2(t)
		Dtype h_minus1_data_2 = get_data_xy(H,num,channels,height,width,n,c,h,w+1); //h_2(t+1)
		Dtype h2_minus1 = g_data_2 * h_minus1_data_2; //g_2(t)*h_2(t+1)

		Dtype g_data_3 = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse); //g_3(t)
		Dtype h_minus1_data_3 = get_data_xy(H,num,channels,height,width,n,c,h+1,w+1); //h_3(t+1)
		Dtype h3_minus1 = g_data_3 * h_minus1_data_3; //g_3(t)*h_3(t+1)

		Dtype h_hype = (1 - c_data) * (h1_minus1 + h2_minus1 + h3_minus1); //sum(g_i(t)*h_i(t+1)) * (1-c) = (1-c) * (g_1(t)*h_1(t+1)+g_2(t)*h_2(t+1)+g_3(t)*h_3(t+1)) 
		Dtype x_hype = c_data * x_data; //c * x

		Dtype h_data = x_hype + h_hype; //(1-sum(g_i(t))) * x_i(t) + sum(g_i(t) * h_i(t+1))

		set_data_xy(H,num,channels,height,width,n,c,h,w,h_data); //set H data at point x
	}
}
/*END h(t) = c * x_i(t) + (1-c) * sum(g_i(t) * h_i(t+1))*/

/*TOP->BOTTOM*/
/*h(t) = c * x_i(t) + (1-c) * sum(g_i(t) * h_i(t+1))*/
template <typename Dtype>
__global__ void forward_one_row_top_bottom(const int count, int T, int num, int channels, int height, int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, Dtype* H, bool horizontal, bool reverse){
//count -> total number of threads; T -> current_row/current_column; num -> total num batch
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){
		//CUDA kernel loop, index trace the current thread
		int wc_count = width * channels;

		int n,c,h,w; //w->current_col; n->current_batch; c->current_channel;
		int temp = index; 
		h = T;
		n = temp / wc_count;
		temp = temp % wc_count;
		c = temp / width;
		temp = temp % width;
		w = temp;
		//locate the pixel as (n,c,h,w);

		//modify logic by xueyan
		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w); //x
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c
	
		Dtype g_data_1 = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse); //g_(t)1
		Dtype h_minus1_data_1 = get_data_xy(H,num,channels,height,width,n,c,h-1,w-1); //h_(t-1)1
		Dtype h1_minus1 = g_data_1 * h_minus1_data_1; //g_(t)1 * h_(t-1)1

		Dtype g_data_2 = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse); //g_(t)2
		Dtype h_minus1_data_2 = get_data_xy(H,num,channels,height,width,n,c,h-1,w); //h_(t-1)2
		Dtype h2_minus1 = g_data_2 * h_minus1_data_2; //g_(t)2 * h_(t-1)2

		Dtype g_data_3 = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse); //g_(t)3
		Dtype h_minus1_data_3 = get_data_xy(H,num,channels,height,width,n,c,h-1,w+1); //h_(t-1)3
		Dtype h3_minus1 = g_data_3 * h_minus1_data_3; //g_(t)3 * h_(t-1)3

		Dtype h_hype = (h1_minus1 + h2_minus1 + h3_minus1) * (1 - c_data); //(1-c)*(sum(g_(t)i * h_(t-1)i)) = (1-c) * (g_(t)1*h_(t-1)1+g_(t)2*h_(t-1)2+g_(t)3*h_(t-1)3)
		Dtype x_hype = c_data * x_data; //c * x_(t)i

		Dtype h_data = x_hype + h_hype; //(1-sum(g_(t)i)) * x_(t)i + sum(g_(t)i * h_(t-1)i)

		set_data_xy(H,num,channels,height,width,n,c,h,w,h_data);
	}
}
/*END h(t) = c * x_i(t) + (1-c) * sum(g_i(t) * h_i(t+1))*/

/*BOTTOM->TOP*/
/*h(t) = c * x_i(t) + (1-c) * sum(g_i(t) * h_i(t+1))*/
template <typename Dtype>
__global__ void forward_one_row_bottom_top(const int count, int T, int num, int channels, int height,  int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, Dtype* H, bool horizontal, bool reverse){
//count -> total number of threads; T -> current_row/current_column; num -> total num batch
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){
		//CUDA kernel loop, index trace the current thread
		int wc_count = width * channels;

		int n,c,h,w; //w->current_col; n->current_batch; c->current_channel;
		int temp = index;
		h = T;
		n = temp / wc_count;
		temp = temp % wc_count;
		c = temp / width;
		temp = temp % width;
		w = temp;
		//locate the pixel as (n,c,h,w);

		//modify logic by xueyan
		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w); //w
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c
 
		Dtype g_data_1 = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse); //g_(t)1
		Dtype h_minus1_data_1 = get_data_xy(H,num,channels,height,width,n,c,h+1,w-1); //h_(t+1)1
		Dtype h1_minus1 = g_data_1 * h_minus1_data_1; //g_(t)1 * h_(t+1)1

		Dtype g_data_2 = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse); //g_(t)2
		Dtype h_minus1_data_2 = get_data_xy(H,num,channels,height,width,n,c,h+1,w); //h_(t+1)2
		Dtype h2_minus1 = g_data_2 * h_minus1_data_2; //g_(t)2 * h_(t+1)2

		Dtype g_data_3 = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse); //g_(t)3
		Dtype h_minus1_data_3 = get_data_xy(H,num,channels,height,width,n,c,h+1,w+1); //h_(t+1)3
		Dtype h3_minus1 = g_data_3 * h_minus1_data_3; //g_(t)3 * h_(t+1)3

		Dtype h_hype = (1 - c_data) * (h1_minus1 + h2_minus1 + h3_minus1); //(1-c)*sum(g_(t)i * h_(t+1)i) = (1-c)*(g_(t)1*h_(t+1)1+g_(t)2*h_(t+1)2+g_(t)3*h_(t+1)3)
		Dtype x_hype = c_data * x_data; //c * x_(t)i

		Dtype h_data = x_hype + h_hype; //(1-sum(g_(t)i)) * x_(t)i + sum(g_(t)i * h_(t+1)i)

		set_data_xy(H,num,channels,height,width,n,c,h,w,h_data);
	}
}
/*END h(t) = c * x_i(t) + (1-c) * sum(g_i(t) * h_i(t+1))*/


template <typename Dtype>
__global__ void backward_one_col_left_right(const int count, int T, int num, int channels, int height, int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, const Dtype* H, Dtype* X_diff, Dtype* G1_diff, Dtype* G2_diff, Dtype* G3_diff, Dtype* C_diff, Dtype* H_diff, bool horizontal, bool reverse){
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){ 
		//CUDA kernel loop, index trace the current thread
		int hc_count = height * channels;

		int n,c,h,w; //w->current_col; n->current_batch; c->current_channel;
		int temp=index;
		w = T;
		n = temp / hc_count;
		temp = temp % hc_count;
		c = temp / height;
		temp = temp % height;
		h = temp;
		//locate the pixel as (n,c,h,w);

		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w);
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c

		//h(t)_diff = top(t)_diff
		Dtype h_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h,w);

		//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
		Dtype add1_h3_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h-1,w+1);
		Dtype add1_g3_data = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
		Dtype add1_c3_data = get_data_xy(C,num,channels,height,width,n,c,h-1,w+1); //c

		Dtype add1_h2_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h,w+1);
		Dtype add1_g2_data = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
		Dtype add1_c2_data = get_data_xy(C,num,channels,height,width,n,c,h,w+1);

		Dtype add1_h1_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h+1,w+1);
		Dtype add1_g1_data = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
 		Dtype add1_c1_data = get_data_xy(C,num,channels,height,width,n,c,h+1,w+1);

		h_diff = h_diff + add1_h3_diff * add1_g3_data * (1 - add1_c3_data) + add1_h2_diff * add1_g2_data * (1 - add1_c2_data) + add1_h1_diff * add1_g1_data * (1 - add1_c1_data);
	
		//H_diff[n*channels*height*width + c*height*width + h*width + w]=0;
		set_data_xy(H_diff,num,channels,height,width,n,c,h,w,h_diff); 

		//x(t)_diff=(1-sum(g_date))*h(t)_diff
    	Dtype g1_data =  get_gate_xy(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
		Dtype g2_data =  get_gate_xy(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
		Dtype g3_data =  get_gate_xy(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
	
		Dtype x_diff = c_data * h_diff;
		set_data_xy(X_diff,num,channels,height,width,n,c,h,w,x_diff);

		// g_diff = h_diff * (h_data(t-1) - x_data)
		Dtype h1_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h-1,w-1); 
		Dtype g1_diff = h_diff * h1_minus1_data * (1 - c_data);
		set_gate_xy(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,g1_diff,horizontal,reverse);

		Dtype h2_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h,w-1); 
		Dtype g2_diff = h_diff * h2_minus1_data * (1 - c_data);
		set_gate_xy(G2_diff,num,channels,height,width,n,c,h,w,h,w-1,g2_diff,horizontal,reverse);

		Dtype h3_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h+1,w-1); 
		Dtype g3_diff = h_diff * h3_minus1_data * (1 - c_data);
		set_gate_xy(G3_diff,num,channels,height,width,n,c,h,w,h+1,w-1,g3_diff,horizontal,reverse);

		// c_diff = h_diff * (x - h1_minus1_data - h2_minus1_data - h3_minus1_data)
		Dtype c_diff = h_diff * (x_data - h1_minus1_data*g1_data - h2_minus1_data*g2_data - h3_minus1_data*g3_data);
		set_data_xy(C_diff,num,channels,height,width,n,c,h,w,c_diff);
	}
}
 
template <typename Dtype>
__global__ void backward_one_col_right_left(const int count, int T, int num,int channels, int height, int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, const Dtype* H, Dtype* X_diff, Dtype* G1_diff, Dtype* G2_diff, Dtype* G3_diff, Dtype* C_diff, Dtype* H_diff, bool horizontal, bool reverse){
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){ 
		//CUDA kernel loop, index trace the current thread
		int hc_count = height * channels;

		int n,c,h,w; //w->current_col; n->current_batch; c->current_channel;
		int temp=index;
		w = T;
		n = temp / hc_count;
		temp = temp % hc_count;
		c = temp / height;
		temp = temp % height;
		h = temp;
		//locate the pixel as (n,c,h,w);

		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w);
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c	

		//h(t)_diff = top(t)_diff
		Dtype h_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h,w); 

		//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
		Dtype add1_h3_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h-1,w-1);
		Dtype add1_g3_data = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
		Dtype add1_c3_data = get_data_xy(C,num,channels,height,width,n,c,h-1,w-1);

		Dtype add1_h2_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h,w-1);
		Dtype add1_g2_data = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h,w-1,horizontal,reverse);
		Dtype add1_c2_data = get_data_xy(C,num,channels,height,width,n,c,h,w-1);

		Dtype add1_h1_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h+1,w-1);
		Dtype add1_g1_data = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
		Dtype add1_c1_data = get_data_xy(C,num,channels,height,width,n,c,h+1,w-1);

		h_diff = h_diff + add1_h3_diff * add1_g3_data * (1 - add1_c3_data) + add1_h2_diff * add1_g2_data * (1 - add1_c2_data) + add1_h1_diff * add1_g1_data * (1 - add1_c1_data);
		set_data_xy(H_diff,num,channels,height,width,n,c,h,w,h_diff); 

    	Dtype g1_data = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
		Dtype g2_data = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h,w+1,horizontal,reverse);
		Dtype g3_data = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);

		Dtype x_diff = c_data * h_diff;
		set_data_xy(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	
    	// g_diff = h_diff * (h_data(t-1) - x_data)
		Dtype h1_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h-1,w+1); 
		Dtype g1_diff = h_diff * h1_minus1_data * (1 - c_data);
		set_gate_xy(G1_diff,num,channels,height,width,n,c,h,w,h-1,w+1,g1_diff,horizontal,reverse);

		Dtype h2_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h,w+1); 
		Dtype g2_diff = h_diff * h2_minus1_data * (1 - c_data);
		set_gate_xy(G2_diff,num,channels,height,width,n,c,h,w,h,w+1,g2_diff,horizontal,reverse);

		Dtype h3_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h+1,w+1); 
		Dtype g3_diff = h_diff * h3_minus1_data * (1 - c_data);
		set_gate_xy(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,g3_diff,horizontal,reverse);

		// c_diff = h_diff * (x - h1_minus1_data - h2_minus1_data - h3_minus1_data)
		Dtype c_diff = h_diff * (x_data - h1_minus1_data*g1_data - h2_minus1_data*g2_data - h3_minus1_data*g3_data);
		set_data_xy(C_diff,num,channels,height,width,n,c,h,w,c_diff);
	}
}


template <typename Dtype>
__global__ void backward_one_row_top_bottom(const int count, int T, int num,int channels, int height,  int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, const Dtype* H, Dtype* X_diff, Dtype* G1_diff, Dtype* G2_diff, Dtype* G3_diff, Dtype* C_diff, Dtype* H_diff, bool horizontal, bool reverse){
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){ 
	//CUDA kernel loop, index trace the current thread	
		int wc_count = width * channels;

		int n,c,h,w; //w->current_col; n->current_batch; c->current_channel;
		int temp=index;
		h = T;
		n = temp / wc_count;
		temp = temp % wc_count;
		c = temp / width;
		temp = temp % width;
		w = temp;
		//locate the pixel as (n,c,h,w);

		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w);
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c
		Dtype h_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h,w); 

		//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
		Dtype add1_h3_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h+1,w-1);
		Dtype add1_g3_data = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
		Dtype add1_c3_data = get_data_xy(C,num,channels,height,width,n,c,h+1,w-1); 

		Dtype add1_h2_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h+1,w);
		Dtype add1_g2_data = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
		Dtype add1_c2_data = get_data_xy(C,num,channels,height,width,n,c,h+1,w);

		Dtype add1_h1_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h+1,w+1);
		Dtype add1_g1_data = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
		Dtype add1_c1_data = get_data_xy(C,num,channels,height,width,n,c,h+1,w+1);

		h_diff = h_diff + add1_h3_diff * add1_g3_data * (1 - add1_c3_data) + add1_h2_diff * add1_g2_data * (1 - add1_c2_data) + add1_h1_diff * add1_g1_data * (1 - add1_c1_data);
		set_data_xy(H_diff,num,channels,height,width,n,c,h,w,h_diff); 

		//x(t)_diff=(1-g(t))*h(t)_diff
		Dtype g1_data =  get_gate_xy(G1,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
		Dtype g2_data =  get_gate_xy(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
		Dtype g3_data =  get_gate_xy(G3,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
		Dtype x_diff = c_data * h_diff;
		set_data_xy(X_diff,num,channels,height,width,n,c,h,w,x_diff);
	
		// g_diff = h_diff * (h_data(t-1) - x_data)
		Dtype h1_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h-1,w-1); 
		Dtype g1_diff = h_diff * h1_minus1_data * (1 - c_data);
		set_gate_xy(G1_diff,num,channels,height,width,n,c,h,w,h-1,w-1,g1_diff,horizontal,reverse);

		Dtype h2_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h-1,w); 
		Dtype g2_diff = h_diff * h2_minus1_data * (1 - c_data);
		set_gate_xy(G2_diff,num,channels,height,width,n,c,h,w,h-1,w,g2_diff,horizontal,reverse);

		Dtype h3_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h-1,w+1); 
		Dtype g3_diff = h_diff * h3_minus1_data * (1 - c_data);
		set_gate_xy(G3_diff,num,channels,height,width,n,c,h,w,h-1,w+1,g3_diff,horizontal,reverse);

		// c_diff = h_diff * (x - h1_minus1_data - h2_minus1_data - h3_minus1_data)
		Dtype c_diff = h_diff * (x_data - h1_minus1_data*g1_data - h2_minus1_data*g2_data - h3_minus1_data*g3_data);
		set_data_xy(C_diff,num,channels,height,width,n,c,h,w,c_diff);
	}
}

template <typename Dtype>
__global__ void backward_one_row_bottom_top(const int count, int T, int num, int channels, int height,  int width, const Dtype* X, const Dtype* G1, const Dtype* G2, const Dtype* G3, const Dtype* C, const Dtype* H, Dtype* X_diff, Dtype* G1_diff, Dtype* G2_diff, Dtype* G3_diff, Dtype* C_diff, Dtype* H_diff, bool horizontal, bool reverse){
	for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x * gridDim.y){ 
	//CUDA kernel loop, index trace the current thread
		int wc_count = width * channels;

		int n,c,h,w;
		int temp=index;
		h = T;
		n = temp / wc_count;
		temp = temp % wc_count;
		c = temp / width;
		temp = temp % width;
		w = temp;

		Dtype x_data = get_data_xy(X,num,channels,height,width,n,c,h,w);
		Dtype c_data = get_data_xy(C,num,channels,height,width,n,c,h,w); //c
		//h(t)_diff = top(t)_diff
		Dtype h_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h,w); 

		//h(t)_diff += h(t+1)_diff * g(t+1) if t<T
		Dtype add1_h3_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h-1,w-1);
		Dtype add1_g3_data = get_gate_xy(G3,num,channels,height,width,n,c,h,w,h-1,w-1,horizontal,reverse);
		Dtype add1_c3_data = get_data_xy(C,num,channels,height,width,n,c,h-1,w-1);

		Dtype add1_h2_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h-1,w);
		Dtype add1_g2_data = get_gate_xy(G2,num,channels,height,width,n,c,h,w,h-1,w,horizontal,reverse);
		Dtype add1_c2_data = get_data_xy(C,num,channels,height,width,n,c,h-1,w);

		Dtype add1_h1_diff = get_data_xy(H_diff,num,channels,height,width,n,c,h-1,w+1);
		Dtype add1_g1_data = get_gate_xy(G1,num,channels,height,width,n,c,h,w,h-1,w+1,horizontal,reverse);
		Dtype add1_c1_data = get_data_xy(C,num,channels,height,width,n,c,h-1,w+1);

		h_diff = h_diff + add1_h3_diff * add1_g3_data * (1 - add1_c3_data) + add1_h2_diff * add1_g2_data * (1 - add1_c2_data) + add1_h1_diff * add1_g1_data * (1 - add1_c1_data);
		set_data_xy(H_diff,num,channels,height,width,n,c,h,w,h_diff); 

		//x(t)_diff=(1-g(t))*h(t)_diff
		Dtype g1_data =  get_gate_xy(G1,num,channels,height,width,n,c,h,w,h+1,w-1,horizontal,reverse);
		Dtype g2_data =  get_gate_xy(G2,num,channels,height,width,n,c,h,w,h+1,w,horizontal,reverse);
		Dtype g3_data =  get_gate_xy(G3,num,channels,height,width,n,c,h,w,h+1,w+1,horizontal,reverse);
 		Dtype x_diff = c_data * h_diff;
		set_data_xy(X_diff,num,channels,height,width,n,c,h,w,x_diff);

		// g_diff = h_diff * (h_data(t-1) - x_data)
		Dtype h1_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h+1,w-1); 
		Dtype g1_diff = h_diff * h1_minus1_data * (1 - c_data);
		set_gate_xy(G1_diff,num,channels,height,width,n,c,h,w,h+1,w-1,g1_diff,horizontal,reverse);

		//Dtype g2_diff = h_diff * g2_idx * x_data * -1;
		Dtype h2_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h+1,w); 
		Dtype g2_diff = h_diff * h2_minus1_data * (1 - c_data);
		set_gate_xy(G2_diff,num,channels,height,width,n,c,h,w,h+1,w,g2_diff,horizontal,reverse);      

		//Dtype g3_diff = h_diff * g3_idx * x_data * -1;
		Dtype h3_minus1_data = get_data_xy(H,num,channels,height,width,n,c,h+1,w+1); 
		Dtype g3_diff = h_diff * h3_minus1_data * (1 - c_data);
		set_gate_xy(G3_diff,num,channels,height,width,n,c,h,w,h+1,w+1,g3_diff,horizontal,reverse);

		// c_diff = h_diff * (x - h1_minus1_data - h2_minus1_data - h3_minus1_data)
		Dtype c_diff = h_diff * (x_data - h1_minus1_data*g1_data - h2_minus1_data*g2_data - h3_minus1_data*g3_data);
		set_data_xy(C_diff,num,channels,height,width,n,c,h,w,c_diff);
	}
}

template<typename Dtype>
inline void SCNForward(const Tensor<gpu, 4, Dtype> &data,
					   const Tensor<gpu, 4, Dtype> &g1,
					   const Tensor<gpu, 4, Dtype> &g2,
					   const Tensor<gpu, 4, Dtype> &g3,
					   const Tensor<gpu, 4, Dtype> &c,
					   const Tensor<gpu, 4, Dtype> &out,
					   const bool horizontal_,
					   const bool reverse_){

/*get pointer*/	
	const Dtype *X = data.dptr_;
	const Dtype *G1 = g1.dptr_;
	const Dtype *G2 = g2.dptr_;
	const Dtype *G3 = g3.dptr_;
	const Dtype *C = c.dptr_;
	Dtype *H = out.dptr_;
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

/*allocate kernel*/
	if(horizontal_ && !reverse_){  // left to right
		/*logic within this block:
		 *1. calculate total number of execution units that run in parallel
		 *2. calculate block and grid dimension
		 *3. check block/grid dimension, get stream
		 *4. call cuda kernal function*/
		const int n_operation_parallel = height * n_channel * n_batch;
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

		for(int current_col = 0; current_col < width; current_col++){ //iterate through the column
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN left->right forward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
			forward_one_col_left_right<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_col, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, horizontal_, reverse_);
		}
	}else if(horizontal_ && reverse_){ // right to left
		/*logic same as previous*/
		const int n_operation_parallel = height * n_channel * n_batch; //total number of execution units that run in parallel
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

		for(int current_col = width - 1; current_col >= 0; current_col--){
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN right->left forward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
			forward_one_col_right_left<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_col, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, horizontal_, reverse_);
		}
	}else if(!horizontal_ && !reverse_){ // top to bottom
		/*logic same as previous*/
		const int n_operation_parallel = width * n_channel * n_batch; //total number of execution units that run in parallel
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

		for(int current_row = 0; current_row < height; current_row++){
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN top->bottom forward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_

			forward_one_row_top_bottom<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_row, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, horizontal_, reverse_);
		}
	}else{  //bottom to top
		/*logic same as previous*/
		const int n_operation_parallel = width * n_channel * n_batch; //total number of execution units that run in parallel
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;		

		for(int current_row = height - 1; current_row >= 0; current_row--){
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN bottom->top forward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
	
			forward_one_row_bottom_top<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_row, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, horizontal_, reverse_);
		}
	}
/*END allocate kernel*/

}//end SCNForward

template<typename Dtype>
inline void SCNBackward(const Tensor<gpu, 4, Dtype> &data,
					    const Tensor<gpu, 4, Dtype> &g1,
					    const Tensor<gpu, 4, Dtype> &g2,
					    const Tensor<gpu, 4, Dtype> &g3,
					    const Tensor<gpu, 4, Dtype> &c,
					    const Tensor<gpu, 4, Dtype> &out,
					    const Tensor<gpu, 4, Dtype> &data_diff,
					    const Tensor<gpu, 4, Dtype> &g1_diff,
					    const Tensor<gpu, 4, Dtype> &g2_diff,
					    const Tensor<gpu, 4, Dtype> &g3_diff,
					    const Tensor<gpu, 4, Dtype> &c_diff,
					    const Tensor<gpu, 4, Dtype> &out_diff,
					    const bool horizontal_,
					    const bool reverse_){

/*get pointer*/	
	const Dtype *X = data.dptr_;
	const Dtype *G1 = g1.dptr_;
	const Dtype *G2 = g2.dptr_;
	const Dtype *G3 = g3.dptr_;
	const Dtype *C = c.dptr_;
	const Dtype *H = out.dptr_;

	Dtype *X_diff = data_diff.dptr_;
	Dtype *G1_diff = g1_diff.dptr_;
	Dtype *G2_diff = g2_diff.dptr_;
	Dtype *G3_diff = g3_diff.dptr_;
	Dtype *C_diff = c_diff.dptr_;
	Dtype *H_diff = out_diff.dptr_;
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

/*allocate kernel*/
	if(horizontal_ && !reverse_){  // left to right
		/*logic within this block:
		 *1. calculate total number of execution units that run in parallel
		 *2. calculate block and grid dimension
		 *3. check block/grid dimension, get stream
		 *4. call cuda kernal function*/
		const int n_operation_parallel = height * n_channel * n_batch;
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

		for(int current_col = width - 1; current_col >= 0; current_col--){ //iterate through the column
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN left->right backward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
			backward_one_col_left_right<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_col, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, X_diff, G1_diff, G2_diff, G3_diff, C_diff, H_diff, horizontal_, reverse_);
		}
	}else if(horizontal_ && reverse_){ // right to left
		/*logic same as previous*/
		const int n_operation_parallel = height * n_channel * n_batch; //total number of execution units that run in parallel
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

		for(int current_col = 0; current_col < width; current_col++){
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN right->left backward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
			backward_one_col_right_left<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_col, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, X_diff, G1_diff, G2_diff, G3_diff, C_diff, H_diff, horizontal_, reverse_);
		}
	}else if(!horizontal_ && !reverse_){ // top to bottom
		/*logic same as previous*/
		const int n_operation_parallel = width * n_channel * n_batch; //total number of execution units that run in parallel
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;

		for(int current_row = height - 1; current_row >= 0; current_row--){
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN top->bottom backward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_

			backward_one_row_top_bottom<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_row, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, X_diff, G1_diff, G2_diff, G3_diff, C_diff, H_diff, horizontal_, reverse_);
		}
	}else{  //bottom to top
		/*logic same as previous*/
		const int n_operation_parallel = width * n_channel * n_batch; //total number of execution units that run in parallel
		const int n_blocks_need = ((n_operation_parallel - 1) / NUM_THREADS_BLOCK) + 1;
		const int n_grids_need = ((n_blocks_need - 1) / NUM_BLOCKS_GRID) + 1;		

		for(int current_row = 0; current_row < width; current_row++){
			dim3 dimGrid(NUM_BLOCKS_GRID, n_grids_need);
			dim3 dimBlock(NUM_THREADS_BLOCK);

			CheckLaunchParam(dimGrid, dimBlock, "SCN bottom->top backward"); //check whether dimGrid or dimBlock is out of range
			cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); //??not sure where to find the definition of stream_
	
			backward_one_row_bottom_top<Dtype><<<dimGrid, dimBlock, 0, stream>>>(n_operation_parallel, current_row, n_batch, n_channel, height, width, X, G1, G2, G3, C, H, X_diff, G1_diff, G2_diff, G3_diff, C_diff, H_diff, horizontal_, reverse_);
		}
	}
/*END allocate kernel*/

}//end SCNBackward




} //namespace cuda
template<typename Dtype>
inline void SCNForward(const Tensor<gpu, 4, Dtype> &data,
					   const Tensor<gpu, 4, Dtype> &g1,
					   const Tensor<gpu, 4, Dtype> &g2,
					   const Tensor<gpu, 4, Dtype> &g3,
					   const Tensor<gpu, 4, Dtype> &c,
					   const Tensor<gpu, 4, Dtype> &out,
					   const bool horizontal,
					   const bool reverse){
	cuda::SCNForward(data, g1, g2, g3, c, out, horizontal, reverse);
}

template<typename Dtype>
inline void SCNBackward(const Tensor<gpu, 4, Dtype> &data,
					    const Tensor<gpu, 4, Dtype> &g1,
					    const Tensor<gpu, 4, Dtype> &g2,
					    const Tensor<gpu, 4, Dtype> &g3,
					    const Tensor<gpu, 4, Dtype> &c,
					    const Tensor<gpu, 4, Dtype> &out,
					    const Tensor<gpu, 4, Dtype> &data_diff,
					    const Tensor<gpu, 4, Dtype> &g1_diff,
					    const Tensor<gpu, 4, Dtype> &g2_diff,
					    const Tensor<gpu, 4, Dtype> &g3_diff,
					    const Tensor<gpu, 4, Dtype> &c_diff,
					    const Tensor<gpu, 4, Dtype> &out_diff,
					    const bool horizontal,
					    const bool reverse){
	cuda::SCNBackward(data, g1, g2, g3, c, out, data_diff, g1_diff, g2_diff, g3_diff, c_diff, out_diff, horizontal, reverse);
}

} //namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ScnParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ScnOp<gpu, DType>(param);
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
