#include "Pool.h"

using namespace GB;

Pool::Pool()
{
}

Pool::~Pool()
{
}

void max_operation(const int32_t &a, int32_t &b)
{
	if (a > b)
		b = a;
}

void avg_operation(const int32_t &a, int32_t &b)
{
	b += a;
}

bool Pool::max_execute(Tensor &input, Tensor &output, PoolParam &param)
{
	fill_value_ = INT32_MIN;
	return execute(input, output, param, max_operation);
}

bool Pool::avg_execute(Tensor &input, Tensor &output, PoolParam &param)
{
	fill_value_ = 0;
	return execute(input, output, param, avg_operation);
}

bool Pool::execute(
	Tensor &input, 
	Tensor &output, 
	PoolParam &param,
	void(*f)(const int32_t &a, int32_t &b))
{
	const TensorShape &input_shape = input.GetShape();
	const int &stride = param.stride;
	const int &kernelSize = param.kernel_shape.w;
	const int pad_right_top = param.padding ? (kernelSize >> 1) : 0;
	const int padding = 0;
	TensorShape output_shape;

	output_shape.h = static_cast<int>(std::ceil(static_cast<float>(
		input_shape.h + 2 * padding - kernelSize) / stride)) + 1;

	output_shape.w = static_cast<int>(std::ceil(static_cast<float>(
		input_shape.w + 2 * padding - kernelSize) / stride)) + 1;

	output_shape.c = input_shape.c;

	output.SetShape(output_shape);

	const int &W = output_shape.w;
	const int &H = output_shape.h;
	const int &C = output_shape.c;
	int xi = 0;
	int yi = 0;
	int yOffset = 0;
	int xOffset = 0;
	int elem = 0;
	int32_t* p_in = NULL;
	int32_t* p_out = NULL;
	std::vector<int32_t> maxValue(C);
	
	for (int xo = 0; xo < W; ++xo)	
	{
		xOffset = xo * stride;		

		for (int yo = 0; yo < H; ++yo)
		{
			
			std::fill(maxValue.begin(), maxValue.end(), fill_value_);
			yOffset = yo * stride;

			for (int ky = 0; ky < kernelSize; ++ky)
			{
				yi = yOffset + ky;
				
				if (yi >= input_shape.h)
					continue;

				for (int kx = 0; kx < kernelSize; ++kx)
				{
					xi = xOffset + kx;

					if (xi >= input_shape.w)
						continue;

					elem = yi * input_shape.w + xi;
					input.GetElement32(elem, 0, p_in);

					for (int c = 0; c < C; ++c)
					{
						f(p_in[c], maxValue[c]);						
					}
				}
			}

			// done kernel iteration
			elem = yo * output_shape.w + xo;
			output.GetElement32(elem, 0, p_out);
			
			for (int c = 0; c < C; ++c)
			{
				ChannelQuantisation &quant = param.quantisation[c];

				Tensor::Quantise(
					maxValue[c],
					quant,
					param.clamp_high,
					param.clamp_low);				

				p_out[c] = maxValue[c];				
			}
		}
	}
	return true;
}
