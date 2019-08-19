#pragma once
#include "Tensor.h"

namespace GB {
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
		float GetMACsPerCycle() { return mMACsPerCycle; }
		float GetCyclesPerMAC() { return mCyclesPerMAC; }
	private:
		uint64_t mCycles;
		uint64_t mMACs;
		float mMACsPerCycle;
		float mCyclesPerMAC;
	};
}

