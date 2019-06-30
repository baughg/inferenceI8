#pragma once
#include "Tensor.h"

namespace GB
{
	typedef struct PoolParam
	{
		PoolParam()
		{
			stride = 1;
			padding = 0;
			clamp_high = 127;
			clamp_low = -128;
		}

		std::vector<ChannelQuantisation> quantisation;
		TensorShape kernel_shape;
		int stride;
		int padding;
		int clamp_high;
		int clamp_low;
	}PoolParam;

	class Pool
	{
	public:
		Pool();
		~Pool();
		bool max_execute(Tensor &input, Tensor &output, PoolParam &param);
		bool avg_execute(Tensor &input, Tensor &output, PoolParam &param);
	private:
		bool execute(
			Tensor &input, 
			Tensor &output, 
			PoolParam &param,
			void(*f)(const int32_t &a, int32_t &b));
		int32_t fill_value_;
	};
}