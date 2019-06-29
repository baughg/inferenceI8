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
	private:
		bool execute(
			Tensor &input_a,
			Tensor &input_b,
			Tensor &output,
			ElopsParam &param,
			void(*f)(const int32_t &a, const int32_t &b, int32_t &c));
	};
}