package com.nn.losses;

import com.nn.dtypes.Tensor;

public class CategoricalCrossentropy extends Loss {

	public CategoricalCrossentropy() {
		// TODO Auto-generated constructor stub
	}
	
	public Tensor forward(Tensor predictions, Tensor targets) {
		// Number of samples in prediction (batch size)
		int samples = predictions.values.length;
		
		// Clip data to prevent divide by 0
		// Clip both sides to prevent bias towards 1
		Tensor clipped = Tensor.clip(predictions, 1E-7f, 1f-1E-7f);
		
		Tensor correct_confidences;
		// Probabilities for categorical target values
		if (targets.values[0].length == 1f) {
			correct_confidences = new Tensor(samples, 1);
			for (int i = 0; i < samples; i++) {
				correct_confidences.values[i][0] = clipped.values[i][(int) targets.values[i][0]];
			}
		}
		// Probabilities for one-hot encoded target values
		else {
			correct_confidences = Tensor.sum(Tensor.multiply(targets, clipped), 1);
		}
		// Negative log likelihood
		return Tensor.multiply(Tensor.log(correct_confidences), -1f);
	}
	
	public Tensor backward(Tensor predictions, Tensor targets) {
		// Number of samples in prediction (batch size)
		int samples = predictions.values.length;
		// Number of outputs per sample
		int outputs = predictions.values[0].length;
		
		// Convert sparse to one-hot
		if (targets.values[0].length == 1f) {
			targets = Tensor.sparseToOneHot(targets, outputs);
		}
		// Calculate gradient
		Tensor input_gradients = Tensor.multiply(Tensor.divideWithChecks(targets, predictions), -1f);
		// Normalize gradient
		input_gradients = Tensor.multiply(input_gradients, 1f/samples);
		
		return input_gradients;
	}
}
