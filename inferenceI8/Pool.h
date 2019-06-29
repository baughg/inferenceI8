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
		}

		TensorShape kernel_shape;
		int stride;
		int padding;
	}PoolParam;

	class Pool
	{
	public:
		Pool();
		~Pool();
		bool max_execute(Tensor &input, Tensor &output, PoolParam &param);
	};
}