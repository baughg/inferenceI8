// inferenceI8.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Tensor.h"
#include "Convolution.h"
#include "Pool.h"
#include "Elementwise.h"
#include "MsTimer.h"
using namespace GB;

int main()
{
	GB::Tensor act(NULL,true);
	Tensor wgt;
	TensorShape act_shape(224,224,3,1);
	TensorShape wt_shape(7, 7, 3, 64);

	act.SetShape(act_shape);
	act.FillRand();

	wgt.SetShape(wt_shape);
	wgt.FillRand();

	Convolution conv;
	ConvParam param;
	param.padding = 1;
	param.stride = 2;
	param.quantisation.resize(wt_shape.k);
	Tensor output(NULL,true);
	MsTimer ms_timer;
	ms_timer.start();
	conv.execute(act, wgt, output, param);
	ms_timer.stop();
	
	TensorShape poolKernel(3,3,1,1);
	Tensor pool_output(NULL,true);
	PoolParam pool_param;
	pool_param.stride = 2;
	pool_param.padding = 0;
	pool_param.kernel_shape = poolKernel;
	const int pool_layer_channels = output.GetShape().c;

	pool_param.quantisation.resize(pool_layer_channels);

	for (int c = 0; c < pool_layer_channels; ++c)
	{
		pool_param.quantisation[c].right_shift = 0;
	}

	Pool pool;
	pool.max_execute(output, pool_output, pool_param);

	Elementwise elop;
	ElopsParam elop_param;
	const int elop_layer_channels = pool_output.GetShape().c;
	elop_param.quantisation.resize(elop_layer_channels);

	for (int c = 0; c < elop_layer_channels; ++c)
	{
		elop_param.quantisation[c].right_shift = 0;
	}

	Tensor elop_output(NULL,true);
	elop.add_execute(pool_output, pool_output, elop_output, elop_param);
	printf("inference8: completed in %lld ms.\n", ms_timer.elapsed());
	return 0;
}

