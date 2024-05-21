package com.nn.activations;

import com.nn.dtypes.Tensor;
import com.nn.layers.Layer;

public class ReLU extends Layer {
	
	public ReLU() {
		
		}
	// Forward pass, apply ReLU activation
	public void forward(Tensor inputs) {
		this.inputs = inputs;
		int rows = inputs.values.length;
		int cols = inputs.values[0].length;
		outputs = Tensor.zeros(rows, cols);
		
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				outputs.values[r][c] = Math.max(0f, inputs.values[r][c]);
			}
		}
	}
	
	// Backward pass
	public void backward(Tensor gradients) {
		int rows = gradients.values.length;
		int cols = gradients.values[0].length;
		input_gradients = Tensor.fromFloatArray(gradients.values);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (inputs.values[r][c] <= 0) {
					input_gradients.values[r][c] = 0;
				}
			}
		}
	}
}
