#pragma once
#include "Tensor.h"
#include <immintrin.h>

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
#ifdef _WIN32
		void DotProduct8(
			int32_t* p_inpt,
			int32_t* p_wght_i32,
			const __m256i &mask,
			const __m256i &zero,
			int32_t &reduction);
#endif
		uint64_t mCycles;
		uint64_t mMACs;
		float mMACsPerCycle;
		float mCyclesPerMAC;
	};
}

