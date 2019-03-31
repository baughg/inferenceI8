// inferenceI8.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Tensor.h"
#include "Convolution.h"
using namespace GB;

int main()
{
	GB::Tensor act;
	Tensor wgt;
	TensorShape act_shape(224,224,3,1);
	TensorShape wt_shape(7, 7, 3, 64);

	act.SetShape(act_shape);
	act.FillRand();

	wgt.SetShape(wt_shape);
	wgt.FillRand();

	Convolution conv;
	ConvParam param;
	param.padding = 1;
	param.stride = 2;
	param.quantisation.resize(wt_shape.k);
	Tensor output;
	//conv.execute(act, wgt, output, param);
	return 0;
}

