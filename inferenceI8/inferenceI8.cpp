// inferenceI8.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <omp.h>
#include "stdafx.h"
#include "Tensor.h"
#include "Convolution.h"
#include "Pool.h"
#include "Elementwise.h"
#include "MsTimer.h"
#include "ConvolutionTask.h"
#include "TensorStore.h"
#include "PerformanceCounter.h"

using namespace GB;

int main()
{
	GB::Tensor act(NULL,true);
	Tensor wgt;
	/*TensorShape act_shape(224,224,3,1);
	TensorShape wt_shape(7, 7, 3, 64);*/
	TensorShape act_shape{ 7, 7, 512, 1 };
	TensorShape wt_shape(3, 3, 512, 512);

	act.SetShape(act_shape);
	act.FillRand();

	wgt.SetShape(wt_shape);
	wgt.FillRand();

	ConvParam param;
	param.padding = 1;
	param.stride = 1;
	param.kernel_x = wt_shape.w;
	param.kernel_y = wt_shape.h;
	param.quantisation.resize(wt_shape.k);
	
	TensorStore<int32_t, 16> actI32{};
	TensorStore<int32_t, 16> wgtI32{};
	tensor_convert(from_Tensor<int8_t, 16>(act), actI32);
	tensor_convert(from_Tensor<int8_t, 16>(wgt), wgtI32);
	PerformanceCounter counter{};
	counter.start();
	actI32.reshape_for_compute(param, TensorStore<int32_t, 16>::data);
	actI32.reshape_for_caching();
	
	const uint32_t channel_step{ 16 };
	const uint32_t task_count{ static_cast<uint32_t>(wt_shape.k) };

	std::vector<ConvolutionTask < int32_t, int32_t, channel_step>> tasks{ task_count };
	std::size_t data_size{ 0 };

	for (uint32_t wgt_set{ 0 }; wgt_set < task_count; ++wgt_set) {
		data_size += create_task(
			tasks[wgt_set], actI32, wgtI32, wgt_set, param);
	}

#ifdef _OPENMP	
	int max_threads = 0;
	max_threads = omp_get_max_threads();
	omp_set_num_threads(max_threads);	
#endif 
	for (uint32_t r{ 0 }; r < 100; ++r) {
#pragma omp parallel default(none) shared(task_count,tasks)
		{
#pragma omp for	schedule(dynamic) nowait
			for (int32_t wgt_set{ 0 }; wgt_set < task_count; ++wgt_set) {
				tasks[r].execute();
			}
		}
	}
	counter.stop();
	std::cout << counter << std::endl;

	Convolution conv;	
	Tensor output(NULL,true);
	MsTimer ms_timer;
	ms_timer.start();
	for (uint32_t r{ 0 }; r < 100; ++r) {
		conv.execute(act, wgt, output, param);
	}
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

