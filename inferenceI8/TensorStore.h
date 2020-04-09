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
		enum Type { data, kernel };
		template<typename D1_Ty, typename D2_Ty, std::size_t cstep>
		friend void tensor_convert(
			const TensorStore<D1_Ty, cstep> &t1, 
			TensorStore<D2_Ty, cstep> &t2);

		template<typename D_Ty, std::size_t cstep>
		friend TensorStore<D_Ty, cstep> from_Tensor(const Tensor &t1);
		void reshape_for_compute(const ConvParam &param, Type type);
		void reshape_for_caching();
		bool get_element(const int &elem, const int &k, Data_Ty* &p_data) const;
		void set_stride();
	private:
		using Container = std::vector<Data_Ty>;
		Container data_ {};
		TensorShape shape_ {};
		const uint32_t channel_step_{ chnstep };
		int elements_ {};
		int k_stride_ {};
		Data_Ty* data_ptr_{ nullptr };
	};		
}

#include "TensorStore.hpp"


