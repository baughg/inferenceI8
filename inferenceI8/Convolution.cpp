#include "Convolution.h"
#include <omp.h>
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

using namespace GB;


Convolution::Convolution()
	: mCycles(0),
	mMACs(0)
{
}


Convolution::~Convolution()
{
}

#ifdef _WIN32
inline void Convolution::DotProduct8(
	int32_t* p_inpt,
	int32_t* p_wght_i32,
	const __m256i &mask,
	const __m256i &zero,
	int32_t &reduction)
{
	const __m256i &a = *reinterpret_cast<const __m256i*>(p_inpt);
	const __m256i &b = *reinterpret_cast<const __m256i*>(p_wght_i32);
	__m256i out = _mm256_mullo_epi32(a, b);

	__m256i prod = _mm256_and_si256(out, mask);
	__m256i sumx = _mm256_hadd_epi32(prod, zero);
	__m256i sumx2 = _mm256_hadd_epi32(sumx, zero);
	int32_t* sumx_ptr = reinterpret_cast<int32_t*>(&sumx2);
	reduction += (sumx_ptr[0] + sumx_ptr[4]);
}
#endif

bool Convolution::execute(
	Tensor &input, Tensor &weight, Tensor &output, ConvParam &param)
{
	register uint64_t t1, t2;
	unsigned int ui = 0;
	t1 = __rdtscp(&ui);
#ifdef _OPENMP	
	int max_threads = 0;
	max_threads = omp_get_max_threads();
	omp_set_num_threads(max_threads);
	//#pragma omp parallel
	//	{
	//		int tid = omp_get_thread_num();
	//		printf("Convolution Thread! #%d\n", tid);
	//	}
#endif  

	const TensorShape input_shape = input.GetShape();
	const TensorShape weight_shape = weight.GetShape();

	
	if (param.padding)
		param.padding = weight_shape.w >> 1;


	int width_out = input_shape.w - weight_shape.w + 2 * param.padding;
	width_out /= param.stride;
	width_out++;

	int height_out = input_shape.h - weight_shape.h + 2 * param.padding;
	height_out /= param.stride;
	height_out++;

	TensorShape output_shape(width_out, height_out, weight_shape.k);
	output.SetShape(output_shape);

	const int stride = input_shape.w;

	std::vector<std::vector<int32_t> > accumulator;
	accumulator.resize(weight_shape.k);
	const int output_elements = width_out * width_out;
	mMACs = input_shape.h * input_shape.w * input_shape.c * weight_shape.k;
	mMACs *= (weight_shape.w * weight_shape.h);
	mMACs /= (param.stride*param.stride);
		
#pragma omp parallel default(none) shared(weight_shape,accumulator,input,weight,param)
	{
#pragma omp for	schedule(dynamic) nowait
		for (int k = 0; k < weight_shape.k; ++k)
		{
			int8_t* p_wght = NULL;
			int32_t* p_inpt = NULL;
			int xi = 0;
			int yi = 0;
			int elem = 0;
			int w_elem = 0;
			int out_index = 0;
			int yo_index_offset = 0;
			int yi_index_offset = 0;

			std::vector<int32_t> &accumulate = accumulator[k];
			accumulate.resize(output_elements);
			int32_t* accumulatePtr = &accumulate[0];
#ifdef _WIN32
			__m256i zero = { 0,0,0,0 };
			__m256i mask = _mm256_xor_si256(zero, zero);
			zero = mask;
#endif

			uint32_t* mask_ptr = reinterpret_cast<uint32_t*>(&mask);
			
			int elems_sum = input_shape.c > 8 ? 8 : input_shape.c;

			for (int e = 0; e < elems_sum; ++e)
			{
				mask_ptr[e] = ~0;
			}
			int32_t ygate = 0;
			int32_t xgate = 0;
			int32_t gate = 0;
			std::vector<int32_t> wght(input_shape.c);
			int32_t* p_wght_i32 = &wght[0];
			uint32_t dpnt = 0;

			for (int ky = -param.padding; ky <= param.padding; ++ky) {
				for (int kx = -param.padding; kx <= param.padding; ++kx) {
					w_elem = (ky + param.padding) * weight_shape.w;
					w_elem += (kx + param.padding);

					weight.GetElement(w_elem, k, p_wght);
										
					for (int c = 0; c < input_shape.c; ++c) {
						p_wght_i32[c] = static_cast<int32_t>(p_wght[c]);
					}

					for (int yo = 0; yo < width_out; ++yo)
					{
						yi = param.stride * yo + ky;
						ygate = static_cast<int32_t>(yi < 0 || yi >= input_shape.h) ^ 0x1;
												
						yi *= ygate;
						yo_index_offset = yo * width_out;
						yi_index_offset = yi * stride;

						for (int xo = 0; xo < width_out; ++xo)
						{
							xi = xo * param.stride + kx;
							xgate = static_cast<int32_t>(xi < 0 || xi >= input_shape.w) ^ 0x1;

							xi *= xgate;
							gate = xgate * ygate;
							elem = yi_index_offset + xi;
							out_index = yo_index_offset + xo;
							input.GetElement32(elem, 0, p_inpt);							
							int32_t reduction = 0;
#ifdef _SIMD_KERNEL
							for (int c = 0; c < input_shape.c; c += 8) {
								DotProduct8(
									&p_inpt[c],
									&p_wght_i32[c],
									mask,
									zero,
									reduction);								
							}
#else
							for (int c = 0; c < input_shape.c; ++c) {
								reduction += (p_inpt[c] * p_wght_i32[c]);
							}							
#endif

							reduction *= gate;
							accumulatePtr[out_index] += reduction;
						}
					}
				}
			}

			// Quantisation
			ChannelQuantisation &quant = param.quantisation[k];

			for (int o = 0; o < output_elements; ++o)
			{
				Tensor::Quantise(
					accumulatePtr[o],
					quant,
					param.clamp_high,
					param.clamp_low);
			}
		}
	}


	// Write output	
#pragma omp parallel default(none) shared(accumulator,output,width_out)
	{
#pragma omp for	schedule(dynamic) nowait
		for (int yo = 0; yo < width_out; ++yo)
		{
			int y_offset = yo * width_out;
			int32_t* p_out = NULL;
			int out_index = 0;

			for (int xo = 0; xo < width_out; ++xo)
			{
				out_index = y_offset + xo;
				output.GetElement32(out_index, 0, p_out);

				for (int k = 0; k < output_shape.c; ++k)
				{
					p_out[k] = static_cast<int32_t>(accumulator[k][out_index]);					
				}
			}
		}
	}

	t2 = __rdtscp(&ui) - t1;
	mCycles = t2;
	mMACsPerCycle = static_cast<float>(mMACs) / static_cast<float>(mCycles);
	mCyclesPerMAC = 1.0f / mMACsPerCycle;
	return true;
}
