
namespace GB {
	template<typename D_Ty, typename Acc_Ty, std::size_t chnstep>
	std::size_t create_task(
		ConvolutionTask<D_Ty, Acc_Ty, chnstep> &task,
		const TensorStore<D_Ty, chnstep> &data,
		const TensorStore<D_Ty, chnstep> &kernel,
		const uint32_t &output_channel,
		const ConvParam &param) {
		const auto data_shape{ data.GetShape() };
		const auto kernel_shape{ kernel.GetShape() };
		const auto kernel_size{ kernel_shape.w * kernel_shape.h * kernel_shape.c };
		task.shape_ = data_shape;			
		task.data_ = data.get_data_pointer();
		task.compute_steps_ = data.compute_steps();
		kernel.get_element(0, output_channel, task.kernel_);		
		task.quantisation_ = param.quantisation[output_channel];
		task.stride_ = param.stride;
		task.padding_ = param.padding;
		task.clamp_low_ = param.clamp_low;
		task.clamp_high_ = param.clamp_high;
		auto elements{ task.shape_.h * task.shape_.w };
		elements = ((elements + 1) >> 1) << 1;
		task.accumulator_.resize(elements);
		return 0;
	}

	template<typename Data_Ty, typename Accumulator_Ty, std::size_t chnstep>
	void ConvolutionTask<Data_Ty, Accumulator_Ty, chnstep>::execute() {
		const auto elements{ shape_.h * shape_.w };
		//accumulator_.resize(elements);
				
		Accumulator_Ty* acc_ptr{ accumulator_.data() };
		const auto C{ shape_.c };
		Data_Ty* data_ptr{ data_ };
		Data_Ty* data1_ptr{ data_ + C };
		const auto c_stride{ C << 1 };

		for (auto se{ 0 }; se < elements; se += 2) {			
			auto k_ptr{ kernel_ };
			Accumulator_Ty acc{};
			Accumulator_Ty acc1{};

			for (int e{ 0 }; e < C; ++e) {
				acc += data_ptr[e] * k_ptr[e];
				acc1 += data1_ptr[e] * k_ptr[e];
			}
			
			data_ptr += c_stride;
			data1_ptr += c_stride;
			*acc_ptr++ += acc;
			*acc_ptr++ += acc1;
		}


	}
}