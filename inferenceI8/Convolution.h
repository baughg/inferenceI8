#pragma once
#include "Tensor.h"

namespace GB {
	typedef struct ChannelQuantisation
	{
		ChannelQuantisation()
		{
			scale = 1;
			bias = 0;
			right_shift = 14;
		}
		int scale;
		int bias;
		int right_shift;
	}ChannelQuantisation;

	typedef struct ConvParam
	{
		ConvParam()
		{
			stride = 1;
			padding = 0;
			clamp_high = 127;
			clamp_low = -128;
		}

		std::vector<ChannelQuantisation> quantisation;
		int stride;
		int padding;
		int clamp_high;
		int clamp_low;
	}ConvParam;

	class Convolution
	{
	public:
		Convolution();
		~Convolution();
		bool execute(Tensor &input, Tensor &weight, Tensor &output, ConvParam &param);
	private:

	};
}

