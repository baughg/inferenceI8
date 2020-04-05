#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include "Tensor.h"

namespace GB {	
	template<typename Data_Ty, typename Accumulator_Ty, std::size_t chnstep>
	class ConvolutionTask
	{	
	public:
		template<typename D_Ty, typename Acc_Ty, std::size_t cstep>
		friend void create_task(
			ConvolutionTask<D_Ty, Acc_Ty, cstep> &task,
			const Tensor &data,
			const Tensor &kernel,
			const uint32_t &output_channel);
	private:
		
		using Container = std::vector<Data_Ty>;
		Container data_ {};
		Container kernel_ {};
		const uint32_t channel_step_ { chnstep };
	};
}

#include "Convolution.hpp"