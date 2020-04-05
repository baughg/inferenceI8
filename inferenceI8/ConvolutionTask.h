#pragma once
#include <vector>
#include <cstdint>

namespace GB {
	template<typename Data_Ty, typename Accumulator_Ty, uint32_t chnstep>
	class ConvolutionTask
	{
	public:
		
	private:
		using Container = std::vector<Data_Ty>;
		Container data_ {};
		const uint32_t channel_step_ { chnstep };
	};
}

