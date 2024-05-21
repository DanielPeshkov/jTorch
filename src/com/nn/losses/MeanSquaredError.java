package com.nn.losses;

import com.nn.dtypes.Tensor;

public class MeanSquaredError extends Loss {
	
	public MeanSquaredError() {
		// TODO Auto-generated constructor stub
	}
	
	public Tensor forward(Tensor predictions, Tensor targets) {
		return Tensor.mean(Tensor.power(Tensor.subtract(targets, predictions), 2f), 1);
	}
	
	public Tensor backward(Tensor predictions, Tensor targets) {
		// Number of samples in prediction (batch size)
		int samples = predictions.values.length;
		// Number of outputs per sample
		int outputs = predictions.values[0].length;
		
		// Gradient on values
		Tensor input_gradients = Tensor.divide(Tensor.subtract(targets, predictions), -2f*outputs);
		// Normalize gradient
		input_gradients = Tensor.divide(input_gradients, samples);
		
		return input_gradients;
	}
}
