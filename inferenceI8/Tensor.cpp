#include "Tensor.h"
#include <cmath>

using namespace GB;

Tensor::Tensor(int8_t* data_ptr, bool i32word)
	: data_ptr_(data_ptr),
	i32_word_(i32word)
{
	data32_ptr_ = reinterpret_cast<int32_t*>(data_ptr_);
}


Tensor::~Tensor()
{
}

void Tensor::SetShape(TensorShape shape)
{
	shape_ = shape;
	elements_ = shape_.w * shape_.h;
	k_stride_ = elements_ * shape_.c;

	int points = k_stride_ * shape_.k;
	
	if (i32_word_)
		points <<= 2;

	if (!data_ptr_) {
		data_.resize(points);
		data_ptr_ = &data_[0];
		data32_ptr_ = reinterpret_cast<int32_t*>(data_ptr_);
	}
}

void Tensor::FillRand()
{
	const size_t data_size = data_.size();
	int v = 0;
	int8_t &vI8 = *reinterpret_cast<int8_t*>(&v);

	

	if (i32_word_)
	{
		const size_t data_points = data_size >> 2;
		int32_t* p_data = reinterpret_cast<int32_t*>(&data_[0]);
		int rnd = 0;
		bool neg = false;

		for (size_t p = 0; p < data_points; ++p)
		{
			rnd = rand() - (RAND_MAX >> 1);
			neg = rnd < 0;
			rnd &= 0xff;

			if (neg)
				rnd *= -1;

			p_data[p] = rnd;
		}
	}
	else
	{
		for (size_t p = 0; p < data_size; ++p) {
			v = rand() % 256;
			data_[p] = vI8;
		}
	}
}

TensorShape Tensor::GetShape() const
{
	return shape_;
}

void Tensor::Quantise(
	int32_t &q, 
	const ChannelQuantisation &quant,
	const int &clamp_high,
	const int &clamp_low)
{
	q += quant.bias;
	q *= quant.scale;

	int64_t scalem_stage_ext = static_cast<int64_t>(q) << 1;
	int64_t round_stage = scalem_stage_ext >> quant.right_shift;
	int64_t round_stage_div = 0;
	round_stage_div = (round_stage == -1) ? 0LL : round_stage;

	int64_t scalem_stage_trunc = round_stage_div >> 1;
	q = static_cast<int32_t>(scalem_stage_trunc);


	if (q > clamp_high)
		q = clamp_high;
	else if (q < clamp_low)
		q = clamp_low;
}

bool Tensor::GetElement(const int &elem, const int &k,int8_t* &p_data)
{
	p_data = NULL;

	if (elem >= elements_ || elem < 0 || k >= shape_.k)
		return false;

	p_data = data_ptr_;
	p_data += elem * shape_.c;
	p_data += k * k_stride_;
	return true;
}

bool Tensor::GetElement32(const int &elem, const int &k, int32_t* &p_data)
{
	p_data = NULL;

	if (elem >= elements_ || elem < 0 || k >= shape_.k)
		return false;

	p_data = data32_ptr_;
	p_data += elem * shape_.c;
	p_data += k * k_stride_;
	return true;
}