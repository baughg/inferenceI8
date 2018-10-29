#include "Tensor.h"
#include <cmath>

using namespace GB;

Tensor::Tensor()
{
}


Tensor::~Tensor()
{
}

void Tensor::SetShape(TensorShape shape)
{
	shape_ = shape;
	elements_ = shape_.w * shape_.h;
	k_stride_ = elements_ * shape_.c;

	const int points = k_stride_ * shape_.k;		
	data_.resize(points);
}

void Tensor::FillRand()
{
	const size_t data_size = data_.size();
	int v = 0;
	int8_t &vI8 = *reinterpret_cast<int8_t*>(&v);

	for (size_t p = 0; p < data_size; ++p) {
		v = rand() % 256;
		data_[p] = vI8;
	}
}

TensorShape Tensor::GetShape() const
{
	return shape_;
}


bool Tensor::GetElement(const int &elem, const int &k,int8_t* &p_data)
{
	p_data = NULL;

	if (elem >= elements_ || elem < 0 || k >= shape_.k)
		return false;

	p_data = &data_[0];
	p_data += elem * shape_.c;
	p_data += k * k_stride_;
	return true;
}