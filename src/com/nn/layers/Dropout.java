package com.nn.layers;

import com.nn.dtypes.Tensor;

public class Dropout extends Layer {
	float rate;
	Tensor mask;

	public Dropout(float rate) {
		this.rate = rate;
	}
	
	public void forward(Tensor inputs) {
		// Output size
		int rows = inputs.values.length;
		int cols = inputs.values[0].length;
		
		// Generate new mask
		mask = Tensor.binaryMask(rate, rows, cols);
		// Apply mask to inputs
		outputs = Tensor.multiply(inputs, mask);
	}
	
	public void backward(Tensor gradients) {
		input_gradients = Tensor.multiply(gradients, mask);
	}
}
