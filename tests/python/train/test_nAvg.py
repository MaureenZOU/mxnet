import mxnet as mx
import numpy as np

#dim = (n,c,h,w)
highest_dim = 20
dim_num = 20
sample_num = 20

def check_pass_back(x1, x2):
	if (x1-x2)/(x1+x2) < 0.05 or (x1 + x2 == 0.0):
		print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "..........Pass_Forward..........Pass_Backward")
	else:
		print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "..........Fail_Forward..........Fail_Backward")


for i in range(0, dim_num):
	dim = (np.random.randint(2,highest_dim), np.random.randint(2,highest_dim), np.random.randint(4,highest_dim), np.random.randint(4,highest_dim))
	x = mx.sym.Variable('X')

	nAvg = mx.sym.nAvg(X=x, threshold=0.5, name='nAvg')
	nAvg = mx.sym.slice_axis(nAvg, axis=1, begin=0, end=1)
	mod = mx.mod.Module(nAvg, data_names=['X'], context=[mx.gpu(0)], label_names=None)
	mod.bind(data_shapes=[('X', dim)],inputs_need_grad=True)
	mod.init_params()

	x = 10*np.random.random_sample(dim) - 1

	for j in range(0, sample_num):
		check_dim = (np.random.randint(0,dim[0]-1),np.random.randint(0,dim[1]-1),np.random.randint(1,dim[2]-2),np.random.randint(1,dim[3]-2))
		e = 1e-4
		x_plus = np.copy(x)
		x_minus = np.copy(x)
		x_plus[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] = x_plus[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] + e
		x_minus[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] = x_minus[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] - e

		x_m = mx.nd.array(x)
		x_plus_m = mx.nd.array(x_plus)
		x_minus_m = mx.nd.array(x_minus)

		Batch = mx.io.DataBatch(
						data=[x_m],
						provide_data=[('X', dim)]
					)

		Batch_plus = mx.io.DataBatch(
						data=[x_plus_m],
						provide_data=[('X', dim)]
					)

		Batch_minus = mx.io.DataBatch(
						data=[x_minus_m],
						provide_data=[('X', dim)]
					)

		mod.forward(Batch)

		mod.forward(Batch_plus)
		y_plus = mod.get_outputs()[0].asnumpy()
		mod.forward(Batch_minus)
		y_minus = mod.get_outputs()[0].asnumpy()
		mod.forward(Batch)
		y_gpu = mod.get_outputs()[0].asnumpy()
		#print(y_gpu)
		dim_b = list(dim)
		dim_b[1] = 1
		dim_b = tuple(dim_b)
		mod.backward(out_grads=[mx.nd.ones(dim_b)])
		y_gpu_grad = mod.get_input_grads()[0].asnumpy()[check_dim[0],check_dim[1],check_dim[2],check_dim[3]]
		y_grad = (y_plus - y_minus) / (2*e)
		y_grad = y_grad[check_dim[0],0,check_dim[2],check_dim[3]]
		check_pass_back(y_gpu_grad, y_grad)