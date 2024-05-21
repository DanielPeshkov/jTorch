package com.nn.losses;

import com.nn.dtypes.Tensor;
public class SoftmaxActivationCategoricalCrossentropyLoss {

	public SoftmaxActivationCategoricalCrossentropyLoss() {
		// TODO Auto-generated constructor stub
	}
	
	public static Tensor backward(Tensor predictions, Tensor targets) {
		// Number of samples
		int samples = predictions.values.length;
		
		// Convert labels from one-hot to sparse
		if (targets.values[0].length != 1) {
			targets = Tensor.argmax(targets);
		}
		
		// Copy values
		Tensor input_gradients = new Tensor(predictions.values.length, predictions.values[0].length);
		input_gradients.values = predictions.values.clone();
		
		// Calculate gradient
		for (int i = 0; i < samples; i++) {
			input_gradients.values[i][(int) targets.values[i][0]] -= 1;
		}
		// Normalize gradient
		input_gradients = Tensor.multiply(input_gradients, 1f/samples);
		
		return input_gradients;
	}
}
