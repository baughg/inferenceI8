
namespace GB {
	template<typename D_Ty, typename Acc_Ty, std::size_t chnstep>
	void create_task(
		ConvolutionTask<D_Ty, Acc_Ty, chnstep> &task,
		const Tensor &data,
		const Tensor &kernel,
		const uint32_t &output_channel) {
		const auto data_shape{ data.GetShape() };
		const auto kernel_shape{ kernel.GetShape() };
		const auto kernel_size{ kernel_shape.w * kernel_shape.h * kernel_shape.c };

		task.data_.resize(data_shape.DataPoints());
		task.kernel_.resize(kernel_size);

		int8_t* data_ptr{ };
		data.GetElement(0, 0, data_ptr);

		std::memcpy(task.data_.data(), data_ptr, task.data_.size() * sizeof(D_Ty));
	}
}