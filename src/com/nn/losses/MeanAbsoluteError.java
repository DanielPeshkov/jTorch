package com.nn.losses;

import com.nn.dtypes.Tensor;

public class MeanAbsoluteError extends Loss {

	public MeanAbsoluteError() {
		// TODO Auto-generated constructor stub
	}
	
	public Tensor forward(Tensor predictions, Tensor targets) {
		return Tensor.mean(Tensor.abs(Tensor.subtract(targets, predictions)), 1);
	}
	
	public Tensor backward(Tensor predictions, Tensor targets) {
		// Number of samples in prediction (batch size)
		int samples = predictions.values.length;
		// Number of outputs per sample
		int outputs = predictions.values[0].length;
		
		// Gradient on values
		Tensor input_gradients = Tensor.divide(Tensor.sign(Tensor.subtract(targets, predictions)), -1f*outputs);
		// Normalize gradient
		input_gradients = Tensor.divide(input_gradients, samples);
		
		return input_gradients;
	}
}
