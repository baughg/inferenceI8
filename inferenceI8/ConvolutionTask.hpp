
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
		const auto elements{ task.shape_.h * task.shape_.w };
		task.accumulator_.resize(elements);
		return 0;
	}

	template<typename Data_Ty, typename Accumulator_Ty, std::size_t chnstep>
	void ConvolutionTask<Data_Ty, Accumulator_Ty, chnstep>::execute() {
		const auto elements{ shape_.h * shape_.w };
		//accumulator_.resize(elements);
		Data_Ty* data_ptr{ data_ };
		Data_Ty* kernel_ptr{ kernel_ };
		Data_Ty* acc_ptr{};

		for (auto cs{ 0 }; cs < compute_steps_; ++cs) {
			acc_ptr = accumulator_.data();
			for (auto se{ 0 }; se < elements; ++se) {
				auto k_ptr{ kernel_ptr };
				Accumulator_Ty acc{};
				for (int e = 0; e < channel_step_; ++e) {					
					acc += data_ptr[e] * k_ptr[e];					
				}
				data_ptr += channel_step_;
				*acc_ptr++ += acc;
			}
			kernel_ptr += channel_step_;
		}
	}
}