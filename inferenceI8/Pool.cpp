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
	const int pad_right_top = param.padding ? (kernelSize >> 1) : 0;
	const int padding = 0;
	TensorShape output_shape;

	output_shape.h = static_cast<int>(ceil(static_cast<float>(
		input_shape.h + 2 * padding - kernelSize) / stride)) + 1;

	output_shape.w = static_cast<int>(ceil(static_cast<float>(
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
			std::fill(maxValue.begin(), maxValue.end(), INT32_MIN);
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
						if (p_in[c] > maxValue[c])
							maxValue[c] = p_in[c];
					}
				}
			}

			// done kernel iteration
			elem = yo * output_shape.w + xo;
			output.GetElement32(elem, 0, p_out);
			
			for (int c = 0; c < C; ++c)
			{
				ChannelQuantisation &quant = param.quantisation[c];
				maxValue[c] += quant.bias;
				maxValue[c] *= quant.scale;
				maxValue[c] >>= quant.right_shift;

				if (maxValue[c] > param.clamp_high)
					maxValue[c] = param.clamp_high;
				else if (maxValue[c] < param.clamp_low)
					maxValue[c] = param.clamp_low;

				p_out[c] = maxValue[c];				
			}
		}
	}
	return true;
}
