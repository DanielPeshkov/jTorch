package com.nn.activations;

import com.nn.dtypes.Tensor;
import com.nn.layers.Layer;

public class Linear extends Layer {

	public Linear() {
		// TODO Auto-generated constructor stub
	}
	
	public void forward(Tensor inputs) {
		// output shape
		int rows = inputs.values.length;
		int cols = inputs.values[0].length;
		
		// Copy input values to output
		outputs = new Tensor(rows, cols);
		outputs.values = inputs.values.clone();
	}
	
	public void backward(Tensor gradients) {
		// output shape
		int rows = gradients.values.length;
		int cols = gradients.values[0].length;
		
		// Copy gradient values to input_gradients
		input_gradients = new Tensor(rows, cols);
		input_gradients.values = gradients.values.clone();
	}
}
