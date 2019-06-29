#include "Pool.h"

using namespace GB;

Pool::Pool()
{
}

Pool::~Pool()
{
}

bool Pool::max_execute(Tensor &input, Tensor &output, PoolParam &param)
{
	const TensorShape &input_shape = input.GetShape();
	const int &stride = param.stride;
	const int &kernelSize = param.kernel_shape.w;
	int padding = param.padding;
	TensorShape output_shape;

	output_shape.h = static_cast<int>(ceil(static_cast<float>(
		input_shape.h + 2 * padding - kernelSize) / stride)) + 1;

	output_shape.w = static_cast<int>(ceil(static_cast<float>(
		input_shape.w + 2 * padding - kernelSize) / stride)) + 1;

	return true;
}
