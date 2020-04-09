namespace GB {
	template<typename Data_Ty, std::size_t chnstep>
	void TensorStore<Data_Ty, chnstep>::reshape_for_caching()
	{
		set_stride();
		auto compute_steps{ shape_.c / channel_step_ };
		const C{ elements_ * channel_step_ };

	}

	template<typename Data_Ty, std::size_t chnstep>
	void TensorStore<Data_Ty, chnstep>::set_stride() {
		elements_ = shape_.w * shape_.h;
		k_stride_ = elements_ * shape_.c;
		data_ptr_ = &data_[0];
	}

	template<typename Data_Ty, std::size_t chnstep>
	bool TensorStore<Data_Ty, chnstep>::get_element(
		const int &elem, const int &k, Data_Ty* &p_data) const
	{
		p_data = nullptr;

		if (elem >= elements_ || elem < 0 || k >= shape_.k)
			return false;

		p_data = data_ptr_;
		p_data += elem * shape_.c;
		p_data += k * k_stride_;
		return true;
	}

	template<typename Data_Ty, std::size_t chnstep>
	void TensorStore<Data_Ty, chnstep>::reshape_for_compute(
		const ConvParam &param, Type type)
	{
		set_stride();

		if (type == kernel) {
			auto C{ shape_.h * shape_.w * shape_.c };
			shape_.c = C;
			shape_.h = 1;
			shape_.w = 1;
			return;
		}

		auto padding_x { 0 };
		auto padding_y { 0 };
		const auto kernel_x { param.kernel_x };
		const auto kernel_y { param.kernel_y };

		if (param.padding) {
			padding_x = kernel_x >> 1;
			padding_y = kernel_y >> 1;
		}

		const auto stride { shape_.w };
		const auto kernel_stride{ param.stride };

		auto width_out = shape_.w - kernel_x + (padding_x << 1);
		width_out /= kernel_stride;
		width_out++;

		auto height_out = shape_.h - kernel_y + (padding_y << 1);
		height_out /= kernel_stride;
		height_out++;

		const auto C { shape_.c * kernel_x * kernel_y };
		TensorStore<Data_Ty, chnstep> output{};
		output.shape_ = TensorShape{ width_out, height_out, C, 1 };
		output.data_.resize(output.shape_.DataPoints());
		output.set_stride();

		auto xi{ 0 };
		auto yi{ 0 };
		auto elem{ 0 };
		auto w_elem{ 0 };
		auto out_index{ 0 };
		auto yo_index_offset{ 0 };
		auto yi_index_offset{ 0 };
		auto ygate{ 0 };
		auto xgate{ 0 };
		auto gate{ 0 };
		Data_Ty* in_ptr {};
		Data_Ty* out_ptr {}; 

		for (auto ky = -padding_y; ky <= padding_y; ++ky) {
			for (auto kx = -padding_x; kx <= padding_x; ++kx) {
				w_elem = (ky + padding_y) * kernel_x;
				w_elem += (kx + padding_x);
				w_elem *= shape_.c;
				
				for (int yo = 0; yo < width_out; ++yo)
				{
					yi = kernel_stride * yo + ky;
					ygate = static_cast<int32_t>(yi < 0 || yi >= shape_.h) ^ 0x1;

					yi *= ygate;
					yo_index_offset = yo * width_out;
					yi_index_offset = yi * stride;

					for (auto xo = 0; xo < width_out; ++xo)
					{
						xi = xo * kernel_stride + kx;
						xgate = static_cast<int32_t>(xi < 0 || xi >= shape_.w) ^ 0x1;

						xi *= xgate;
						gate = xgate * ygate;												

						if (gate) {
							elem = yi_index_offset + xi;
							out_index = yo_index_offset + xo;
							output.get_element(out_index, 0, out_ptr);
							out_ptr += w_elem;
							get_element(elem, 0, in_ptr);
							std::memcpy(out_ptr, in_ptr, shape_.c);
						}						
					}
				}
			}
		}

		this->data_ = std::move(output.data_);
		this->shape_ = std::move(output.shape_);
		set_stride();
	}

	template<typename D1_Ty, typename D2_Ty, std::size_t cstep>
	void tensor_convert(const TensorStore<D1_Ty, cstep> &t1,
		TensorStore<D2_Ty, cstep> &t2) {
		const auto shape{ t1.shape_ };
		const auto points{ shape.DataPoints() };
		t2.shape_ = t1.shape_;		
		t2.data_.resize(t1.data_.size());

		auto d1_iter = std::begin(t1.data_);
		auto d2_iter = std::begin(t2.data_);
		const auto d1_end = std::end(t1.data_);

		for (; d1_iter != d1_end; d1_iter++) {
			*d2_iter = static_cast<D2_Ty>(*d1_iter);
			d2_iter++;
		}
	}

	template<typename D_Ty, std::size_t cstep>
	TensorStore<D_Ty, cstep> from_Tensor(const Tensor &t1) {
		TensorStore<D_Ty, cstep> tensor {};
		tensor.shape_ = t1.GetShape();
		tensor.data_.resize(tensor.shape_.DataPoints());

		if constexpr (sizeof(D_Ty) == sizeof(int8_t)) {
			int8_t* data_ptr{};
			t1.GetElement(0, 0, data_ptr);

			for (auto iter{ std::begin(tensor.data_) };
				iter != std::end(tensor.data_);
				iter++) {
				*iter = static_cast<D_Ty>(*data_ptr);
				data_ptr++;
			}
		}
		else {
			int32_t* data_ptr{};
			t1.GetElement32(0, 0, data_ptr);

			for (auto iter{ std::begin(tensor.data_) };
				iter != std::end(tensor.data_);
				iter++) {
				*iter = static_cast<D_Ty>(*data_ptr);
				data_ptr++;
			}
		}

		return std::move(tensor);
	}
}