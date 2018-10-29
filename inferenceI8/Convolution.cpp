#include "Convolution.h"

using namespace GB;


Convolution::Convolution()
{
}


Convolution::~Convolution()
{
}

bool Convolution::execute(
	Tensor &input, Tensor &weight, Tensor &output, ConvParam &param)
{
	const TensorShape input_shape = input.GetShape();
	const TensorShape weight_shape = weight.GetShape();

	if (param.padding)
		param.padding = weight_shape.w >> 1;


	int width_out = input_shape.w - weight_shape.w + 2 * param.padding;
	width_out /= param.stride;
	width_out++;

	TensorShape output_shape(width_out, width_out, weight_shape.k);
	output.SetShape(output_shape);
	int xi = 0;
	int yi = 0;
	int elem = 0;
	int w_elem = 0;
	const int stride = input_shape.w;
	int8_t* p_wght = NULL;
	int8_t* p_inpt = NULL;
	std::vector<std::vector<int32_t> > accumulator;
	accumulator.resize(weight_shape.k);
	const int output_elements = width_out*width_out;
	int out_index = 0;

	for (int k = 0; k < weight_shape.k; ++k) 
	{
		std::vector<int32_t> &accumulate = accumulator[k];
		accumulate.resize(output_elements);		

		for (int ky = -param.padding; ky <= param.padding; ++ky) {
			for (int kx = -param.padding; kx <= param.padding; ++kx) {
				w_elem = (ky + param.padding) * weight_shape.w;
				w_elem += (kx + param.padding);

				weight.GetElement(w_elem, k, p_wght);

				for (int yo = 0; yo < width_out; ++yo)
				{
					yi = param.stride * yo + ky;

					for (int xo = 0; xo < width_out; ++xo)
					{
						xi = xo * param.stride + kx;

						elem = yi * stride + xi;
						out_index = yo * width_out + xo;

						if (xi >= 0 && xi < input_shape.w && yi >= 0 && yi < input_shape.h)
						{
							input.GetElement(elem, 0, p_inpt);

							for (int c = 0; c < input_shape.c; ++c)
								accumulate[out_index] += (static_cast<int32_t>(p_inpt[c]) * static_cast<int32_t>(p_wght[c]));
						}
					}
				}
			}
		}

		// Quantisation
		ChannelQuantisation &quant = param.quantisation[k];

		for (int o = 0; o < output_elements; ++o)
		{
			accumulate[o] += quant.bias;
			accumulate[o] *= quant.scale;
			accumulate[o] >>= quant.right_shift;

			if (accumulate[o] > param.clamp_high)
				accumulate[o] = param.clamp_high;
			else if(accumulate[o] < param.clamp_low)
				accumulate[o] = param.clamp_low;
		}
	}

	// Write output
	int y_offset = 0;
	int8_t* p_out = NULL;

	for (int yo = 0; yo < width_out; ++yo)
	{
		y_offset = yo * width_out;

		for (int xo = 0; xo < width_out; ++xo)
		{
			out_index = y_offset + xo;
			output.GetElement(out_index, 0, p_out);

			for (int k = 0; k < output_shape.c; ++k)
			{
				p_out[k] = static_cast<int8_t>(accumulator[k][out_index]);
			}
		}
	}

	return true;
}
