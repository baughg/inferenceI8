#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include "Tensor.h"

namespace GB {
	template<typename Data_Ty, std::size_t chnstep>
	class TensorStore
	{
	public:
		template<typename D1_Ty, typename D2_Ty, std::size_t cstep>
		friend void tensor_convert(
			const TensorStore<D1_Ty, cstep> &t1, 
			TensorStore<D2_Ty, cstep> &t2);

		template<typename D_Ty, std::size_t cstep>
		friend TensorStore<D_Ty, cstep> from_Tensor(const Tensor &t1);
	private:
		using Container = std::vector<Data_Ty>;
		Container data_ {};
		TensorShape shape_{};
		const uint32_t channel_step_{ chnstep };
	};		
}

#include "TensorStore.hpp"


