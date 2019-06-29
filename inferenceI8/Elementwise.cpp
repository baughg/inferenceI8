#include "Elementwise.h"

using namespace GB;

Elementwise::Elementwise()
{
}

Elementwise::~Elementwise()
{
}

void add(const int32_t &a, const int32_t &b, int32_t &c)
{
	c = a + b;
}

bool Elementwise::execute(
	Tensor &input_a,
	Tensor &input_b,
	Tensor &output,
	ElopsParam &param,
	void(*f)(const int32_t &a, const int32_t &b, int32_t &c ))
{
	const TensorShape &input_shape_a = input_a.GetShape();
	const TensorShape &input_shape_b = input_b.GetShape();

	if (input_shape_a.w != input_shape_b.w ||
		input_shape_a.h != input_shape_b.h ||
		input_shape_a.c != input_shape_b.c)
		return false;

	output.SetShape(input_shape_a);

	const int elem_count = input_shape_a.w * input_shape_b.h;
	int32_t* p_a = NULL;
	int32_t* p_b = NULL;
	int32_t* p_out = NULL;
	const int Z = input_shape_a.c;

	for (int elem = 0; elem < elem_count; ++elem)
	{
		input_a.GetElement32(elem, 0, p_a);
		input_b.GetElement32(elem, 0, p_b);
		output.GetElement32(elem, 0, p_out);
		int32_t sum = 0;

		for (int z = 0; z < Z; ++z)
		{
			sum = p_a[z] + p_b[z];

			Tensor::Quantise(
				sum,
				param.quantisation[z],
				param.clamp_high,
				param.clamp_low);

			p_out[z] = sum;
		}
	}
	return true;
}

bool Elementwise::add_execute(
	Tensor &input_a,
	Tensor &input_b,
	Tensor &output,
	ElopsParam &param)
{
	return execute(input_a, input_b, output, param, add);
}