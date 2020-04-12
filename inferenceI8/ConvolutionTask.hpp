
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
		task.data_.resize(data_shape.DataPoints());
		task.kernel_.resize(kernel_size);

		D_Ty* data_ptr{ };
		data.get_element(0, 0, data_ptr);

		std::memcpy(
			task.data_.data(), 
			data_ptr, 
			task.data_.size() * sizeof(D_Ty));

		D_Ty* kernel_ptr{ };
		kernel.get_element(0, output_channel, kernel_ptr);

		std::memcpy(
			task.kernel_.data(),
			kernel_ptr,
			task.kernel_.size() * sizeof(D_Ty));

		task.quantisation_ = param.quantisation[output_channel];
		task.stride_ = param.stride;
		task.padding_ = param.padding;
		task.clamp_low_ = param.clamp_low;
		task.clamp_high_ = param.clamp_high;
		return task.data_.size();
	}
}