package com.nn.activations;

import com.nn.dtypes.Tensor;
import com.nn.layers.Layer;

public class Softmax extends Layer {

	public Softmax() {
		// TODO Auto-generated constructor stub
	}
	
	public void forward(Tensor inputs) {
		outputs = Tensor.exp(Tensor.subtract(inputs,  Tensor.max(inputs)));
		outputs = Tensor.divide(outputs, Tensor.sum(outputs, 1));
	}
	
	public void backward(Tensor gradients) {
		// Output shape
		int rows = gradients.values.length;
		int cols = gradients.values[0].length;
		
		// Create empty Tensor
		input_gradients = new Tensor(rows, cols);
		
		// Enumerate outputs and gradients
		Tensor single_output = new Tensor(1, cols);
		Tensor single_gradient = new Tensor(1, cols);
		for (int r = 0; r < rows; r++) {
			single_output.values[0] = outputs.values[r];
			single_gradient.values[0] = gradients.values[r];
			// Calculate Jacobian matrix of the output
			
			Tensor jacobian_matrix = Tensor.subtract(Tensor.diagflat(single_output), Tensor.dot(single_output, single_output.T()));
			// Calculate sample-wise gradient
			input_gradients.values[r] = Tensor.dot(single_gradient, jacobian_matrix).values[0];
		}
	}
}
