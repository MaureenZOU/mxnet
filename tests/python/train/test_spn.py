import mxnet as mx
import numpy as np

horizontal = True
reverse = False
dim_iter = 10
check_iter = 10
test_cpu = True
highest_dim = 20 #if highest_dim > 20, please flag test_cpu = False

def get_data(n,c,i,j,x):

	if i < 0 or i >= height:
		return 0
	elif j < 0 or j >= width:
		return 0
		
	return x[n,c,i,j]

def get_gate(n,c,i1,j1,i2,j2,x):

	if i1 < 0 or i1 >= height:
		return 0
	elif j1 < 0 or j1 >= width:
		return 0
	elif i2 < 0 or i2 >= height:
		return 0
	elif j2 < 0 or j2 >= width:
		return 0
	
	return x[n,c,i1,j1]

def forward_result(dim,x,g1,g2,g3,horizontal,reverse):
	batch = dim[0]
	channel = dim[1]
	height = dim[2]
	width = dim[3]

	if horizontal and (not reverse):
		h = np.ones(dim) 
		h[:,:,:,0] = x[:,:,:,0]
		for j in range(1, width):
			for i in range(0, height):
				for c in range(0, channel):
					for n in range(0, batch):
						h[n,c,i,j] = (1 - get_gate(n,c,i,j,i-1,j-1,g1) - get_gate(n,c,i,j,i,j-1,g2) - get_gate(n,c,i,j,i+1,j-1,g3))*get_data(n,c,i,j,x) + get_gate(n,c,i,j,i-1,j-1,g1)*get_data(n,c,i-j,j-1,h) + get_gate(n,c,i,j,i,j-1,g2)*get_data(n,c,i,j-1,h) + get_gate(n,c,i,j,i+1,j-1,g3)*get_data(n,c,i+1,j-1,h)
		return h
	elif horizontal and reverse:
		h = np.ones(dim) 
		h[:,:,:,width-1] = x[:,:,:,width-1]

		for j in range(width-1, -1, -1):
			for i in range(0, height):
				for c in range(0, channel):
					for n in range(0, batch):
						h[n,c,i,j] = (1 - get_gate(n,c,i,j,i-1,j+1,g1) - get_gate(n,c,i,j,i,j+1,g2) - get_gate(n,c,i,j,i+1,j+1,g3))*get_data(n,c,i,j,x) + get_gate(n,c,i,j,i-1,j+1,g1)*get_data(n,c,i-1,j+1,h) + get_gate(n,c,i,j,i,j+1,g2)*get_data(n,c,i,j+1,h) + get_gate(n,c,i,j,i+1,j+1,g3)*get_data(n,c,i+1,j+1,h)
		return h
	elif (not horizontal) and not reverse:

		h = np.ones(dim) 
		h[:,:,0,:] = x[:,:,0,:]
		for i in range(0, height):
			for j in range(0, width):
				for c in range(0, channel):
					for n in range(0, batch):
						h[n,c,i,j] = (1 - get_gate(n,c,i,j,i-1,j-1,g1) - get_gate(n,c,i,j,i-1,j,g2) - get_gate(n,c,i,j,i-1,j+1,g3))*get_data(n,c,i,j,x) + get_gate(n,c,i,j,i-1,j-1,g1)*get_data(n,c,i-1,j-1,h) + get_gate(n,c,i,j,i-1,j,g2)*get_data(n,c,i-1,j,h) + get_gate(n,c,i,j,i-1,j+1,g3)*get_data(n,c,i-1,j+1,h)
		return h
	else:

			h = np.ones(dim) 
			h[:,:,height-1,:] = x[:,:,height-1,:]

			for i in range(height-1,-1,-1):
				for j in range(0, width):
					for c in range(0, channel):
						for n in range(0, batch):
							h[n,c,i,j] = (1 - get_gate(n,c,i,j,i+1,j-1,g1) - get_gate(n,c,i,j,i+1,j,g2) - get_gate(n,c,i,j,i+1,j+1,g3))*get_data(n,c,i,j,x) + get_gate(n,c,i,j,i+1,j-1,g1)*get_data(n,c,i+1,j-1,h) + get_gate(n,c,i,j,i+1,j,g2)*get_data(n,c,i+1,j,h) + get_gate(n,c,i,j,i+1,j+1,g3)*get_data(n,c,i+1,j+1,h)
			return h

def check_pass(x1, x2, y1, y2):

	if (x1-x2)/(x1+x2) < 0.05:
		if (y1-y2)/(y1+y2) < 0.05:
			print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "     Forward_GPU = " + str(y1) + "   Forward_CPU = " + str(y2) + "..........Pass_Forward..........Pass_Backward")
		else:
			print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "     Forward_GPU = " + str(y1) + "   Forward_CPU = " + str(y2) + "..........Fail_Forward..........Pass_Backward")
	else:
		if (y1-y2)/(y1+y2) < 0.05:
			print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "     Forward_GPU = " + str(y1) + "   Forward_CPU = " + str(y2) + "..........Pass_Forward..........Fail_Backward")
		else:
			print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "     Forward_GPU = " + str(y1) + "   Forward_CPU = " + str(y2) + "..........Fail_Forward..........Fail_Backward")

def check_pass_back(x1, x2):

	if (x1-x2)/(x1+x2) < 0.05:
		print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "..........Pass_Forward..........Pass_Backward")
	else:
		print("GPU_Cal_grad = " + str(x1) + "    " + "Gradient_Check = " + str(x2) + "..........Fail_Forward..........Fail_Backward")



for i in range(0, dim_iter):
	dim = (np.random.randint(2,highest_dim), np.random.randint(2,highest_dim), np.random.randint(4,highest_dim), np.random.randint(4,highest_dim))
	batch = dim[0]
	channel = dim[1]
	height = dim[2]
	width = dim[3]

	x = mx.sym.Variable('X')
	g1 = mx.sym.Variable('G1')
	g2 = mx.sym.Variable('G2')
	g3 = mx.sym.Variable('G3')
	
	spn = mx.sym.SPN(X=x, G1=g1, G2=g2, G3=g3, horizontal=horizontal, reverse=reverse ,name='spn')
	mod = mx.mod.Module(spn, data_names=['X', 'G1', 'G2', 'G3'], context=[mx.gpu(0)], label_names=None)
	mod.bind(data_shapes=[('X', dim),('G1', dim),('G2', dim),('G3', dim)],inputs_need_grad=True)
	mod.init_params()

	x = np.random.random_sample(dim)
	g1 = np.random.random_sample(dim)
	g2 = np.random.random_sample(dim)
	g3 = np.random.random_sample(dim)
	e = 1e-5

	if test_cpu:
		h_py = forward_result(dim,x,g1,g2,g3,horizontal,reverse)
		h_py = np.sum(h_py)

	for j in range(0, check_iter):
		check_dim = (np.random.randint(0,batch-1),np.random.randint(0,channel-1),np.random.randint(1,height-2),np.random.randint(1,width-2))
		check_var = 'g1'

		if check_var == 'g1':
			check_plus = np.copy(g1)
			check_minus = np.copy(g1)
			check = g1
		elif check_var == 'g2':
			check_plus = np.copy(g2)
			check_minus = np.copy(g2)
			check = g2
		elif check_var == 'g3':
			check_plus = np.copy(g3)
			check_minus = np.copy(g3)
			check = g3
		elif check_var == 'x':
			check_plus = np.copy(x)
			check_minus = np.copy(x)
			check = x

		check_plus[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] = check[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] + e
		check_minus[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] = check[check_dim[0],check_dim[1],check_dim[2],check_dim[3]] - e


		x_m = mx.nd.array(x)
		g1_m = mx.nd.array(g1)
		g2_m = mx.nd.array(g2)
		g3_m = mx.nd.array(g3)
		check_m_plus = mx.nd.array(check_plus)
		check_m_minus = mx.nd.array(check_minus)


		if check_var == 'g1':
			Batch = mx.io.DataBatch(
				data=[x_m, g1_m, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_plus = mx.io.DataBatch(
				data=[x_m, check_m_plus, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_minus = mx.io.DataBatch(
				data=[x_m, check_m_minus, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

		elif check_var == 'g2':
			Batch = mx.io.DataBatch(
				data=[x_m, g1_m, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_plus = mx.io.DataBatch(
				data=[x_m, g1_m, check_m_plus, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_minus = mx.io.DataBatch(
				data=[x_m, g1_m, check_m_minus, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)
		elif check_var == 'g3':
			Batch = mx.io.DataBatch(
				data=[x_m, g1_m, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_plus = mx.io.DataBatch(
				data=[x_m, g1_m, g2_m, check_m_plus],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_minus = mx.io.DataBatch(
				data=[x_m, g1_m, g2_m, check_m_minus],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

		elif check_var == 'x':
			Batch = mx.io.DataBatch(
				data=[x_m, g1_m, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_plus = mx.io.DataBatch(
				data=[check_m_plus, g1_m, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

			Batch_minus = mx.io.DataBatch(
				data=[check_m_minus, g1_m, g2_m, g3_m],
				provide_data=[('X', dim), ('G1', dim), ('G2', dim), ('G3', dim)]
			)

		mod.forward(Batch_plus)
		h_plus = mod.get_outputs()[0].asnumpy()
		mod.forward(Batch_minus)
		h_minus = mod.get_outputs()[0].asnumpy()
		h = mod.forward(Batch)
		mod.backward(out_grads=[mx.nd.ones(dim)])
		h_gpu_grad = mod.get_input_grads()[1].asnumpy()[check_dim[0],check_dim[1],check_dim[2],check_dim[3]]

		h_grad = (h_plus - h_minus) / (2*e)
		h_check_grad = np.sum(h_grad)
		
		if test_cpu:
			h_gpu = mod.get_outputs()[0].asnumpy()
			h_gpu = np.sum(h_gpu)
			check_pass(h_gpu_grad, h_check_grad, h_py, h_gpu)
		else:
			check_pass_back(h_gpu_grad, h_check_grad)


