#pragma once
#include <stdint.h>
#include <vector>

namespace GB {
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

		int w;
		int h;
		int c;
		int k;
		TensorLayout layout;
	}TensorShape;

	class Tensor
	{
	public:
		Tensor();
		~Tensor();
		void SetShape(TensorShape shape);
		void FillRand();
		TensorShape GetShape() const;
		bool GetElement(const int &elem, const int &k, int8_t* &p_data);
	private:
		TensorShape shape_;
		std::vector<int8_t> data_;
		int elements_;
		int k_stride_;
	};
}

