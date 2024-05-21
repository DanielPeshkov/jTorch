package com.nn.layers;

import com.nn.dtypes.Tensor;

public class Dense extends Layer {
	
	public int input_features;
	public int output_features;
	
	// Initialization Constructor Method
	public Dense(int input_features, int output_features) {
		
		// Save shapes
		this.input_features = input_features;
		this.output_features = output_features;
		
		// Initialize empty Tensors
		weights = Tensor.random(input_features, output_features);
		biases = Tensor.zeros(1,  output_features);
	}
	
	// Forward Pass
	public void forward(Tensor inputs) {
		// Save inputs
		this.inputs = inputs;
		
		// Calculate dot product
		outputs = Tensor.add(Tensor.dot(inputs, weights), biases);
	}
	
	// Backward Pass
	public void backward(Tensor gradients) {
		weight_gradients = Tensor.dot(inputs.T(), gradients);
		bias_gradients = Tensor.sum(gradients, 0);
		input_gradients = Tensor.dot(gradients, weights.T());
	}
}
