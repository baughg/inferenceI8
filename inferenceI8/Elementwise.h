#pragma once
#include "Tensor.h"

namespace GB
{
	typedef struct ElopsParam
	{
		ElopsParam()
		{			
			clamp_high = 127;
			clamp_low = -128;
		}

		std::vector<ChannelQuantisation> quantisation;			
		int clamp_high;
		int clamp_low;
	}ElopsParam;

	class Elementwise
	{
	public:
		Elementwise();
		~Elementwise();
		bool add_execute(
			Tensor &input_a, 
			Tensor &input_b,
			Tensor &output, 
			ElopsParam &param);
	};
}