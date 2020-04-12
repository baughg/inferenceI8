#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include "TensorStore.h"

namespace GB {	
	template<typename Data_Ty, typename Accumulator_Ty, std::size_t chnstep>
	class ConvolutionTask
	{	
	public:
		template<typename D_Ty, typename Acc_Ty, std::size_t cstep>
		friend std::size_t create_task(
			ConvolutionTask<D_Ty, Acc_Ty, cstep> &task,
			const TensorStore<D_Ty,cstep> &data,
			const TensorStore<D_Ty, cstep> &kernel,
			const uint32_t &output_channel,
			const ConvParam &param);
	private:		
		using Container = std::vector<Data_Ty>;
		Container data_ {};
		Container kernel_ {};
		TensorShape shape_{};
		const uint32_t channel_step_ { chnstep };
		ChannelQuantisation quantisation_ {};
		int stride_ {};
		int padding_ {};
		int clamp_high_ {};
		int clamp_low_ {};
	};
}

#include "ConvolutionTask.hpp"