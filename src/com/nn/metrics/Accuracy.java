package com.nn.metrics;

import com.nn.dtypes.Tensor;

public class Accuracy {

	public Accuracy() {
		// TODO Auto-generated constructor stub
	}
	
	public static float calculate(Tensor predictions, Tensor targets) {
		// Number of samples in prediction (batch size)
		int samples = predictions.values.length;
		
		// Convert predictions from one-hot to sparse categorical
		if (predictions.values[0].length > 1) {
			predictions = Tensor.argmax(predictions);
		}
		// Convert targets from one-hot to sparse categorical
		if (targets.values[0].length > 1) {
			targets = Tensor.argmax(targets);
		}
		// Initialize output variable
		float accuracy = 0f;
		// Compare each prediction against target
		for (int s = 0; s < samples; s++) {
			if (predictions.values[s][0] == targets.values[s][0]) {
				accuracy += 1f;
			}
		}
		// Divide by batch size for mean accuracy
		return accuracy / (float) samples;
	}
}
