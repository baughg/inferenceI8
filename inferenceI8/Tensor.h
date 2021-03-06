#pragma once
#include <stdint.h>
#include <cstddef>
#include <vector>

namespace GB {
	typedef struct {
		int x;
		int y;
	}fm_dim_pair;

	typedef struct
	{
		fm_dim_pair kernel;
		fm_dim_pair stride;
		fm_dim_pair padding;
		fm_dim_pair paddingTopLeft;
		fm_dim_pair paddingBottomRight;
		fm_dim_pair dilation;
	}LayerArgs;

	typedef struct TensorContainer
	{
		TensorContainer()
		{
			rawData = NULL;
		}

		std::vector<size_t> dims;
		void* rawData;
	}TensorContainer;

	typedef struct ChannelQuantisation
	{
		ChannelQuantisation()
		{
			scale = 1;
			bias = 0;
			right_shift = 14;
		}
		int scale;
		int bias;
		int right_shift;
	}ChannelQuantisation;

	typedef enum
	{
		TL_ZM,
		TL_CM
	}TensorLayout;

	typedef struct TensorShape
	{
		TensorShape()
		{
			w = 0;
			h = 0;
			c = 0;
			k = 1;
			layout = TL_ZM;
		}

		TensorShape(int _w, int _h, int _c, int _k = 1)
		{
			w = _w;
			h = _h;
			c = _c;
			k = _k;
			layout = TL_ZM;
		}

		int DataPoints() const {
			return w*h*c*k;
		}

		int w;
		int h;
		int c;
		int k;
		TensorLayout layout;
	}TensorShape;

	class Tensor
	{
	public:
		Tensor(int8_t* data_ptr = NULL, bool i32word = false);
		~Tensor();
		void SetShape(TensorShape shape);
		void FillRand();
		TensorShape GetShape() const;
		bool GetElement(const int &elem, const int &k, int8_t* &p_data) const;
		bool GetElement32(const int &elem, const int &k, int32_t* &p_data);
		static void Quantise(
			int32_t &q,
			const ChannelQuantisation &quant,
			const int &clamp_high,
			const int &clamp_low);
	private:
		TensorShape shape_;
		std::vector<int8_t> data_;
		int8_t* data_ptr_;
		int32_t* data32_ptr_;
		bool i32_word_;
		int elements_;
		int k_stride_;
	};
}

