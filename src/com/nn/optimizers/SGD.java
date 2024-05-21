package com.nn.optimizers;

import com.nn.dtypes.Tensor;
import com.nn.layers.*;

public class SGD extends Optimizer {
	
	float decay;
	float momentum;
	int iterations;

	public SGD(float learning_rate, float decay, float momentum) {
		this.learning_rate = learning_rate;
		this.decay = decay;
		this.momentum = momentum;
		this.current_learning_rate = learning_rate;
		this.iterations = 0;
	}
	
	// Call before parameter updates
	public void preUpdateParams() {
		if (decay != 0f) {
			current_learning_rate = learning_rate * (1f / (1f + decay * iterations));
		}
	}
	
	// Update parameters
	public void updateParams(Layer layer) {
		Tensor weight_updates;
		Tensor bias_updates;
		// SGD with momentum
		if (momentum != 0f) {
			// initialize momentums to 0 if null
			if (layer.weight_momentums == null) {
				layer.weight_momentums = Tensor.zeros(layer.weights.values.length, layer.weights.values[0].length);
				layer.bias_momentums = Tensor.zeros(layer.biases.values.length, layer.biases.values[0].length);
			}
			// Build weight updates with momentum
			weight_updates = Tensor.subtract(Tensor.multiply(layer.weight_momentums, momentum), Tensor.multiply(layer.weight_gradients, current_learning_rate));
			layer.weight_momentums.values = weight_updates.values.clone();
			
			// Build bias updates with momentum
			bias_updates = Tensor.subtract(Tensor.multiply(layer.bias_momentums, momentum), Tensor.multiply(layer.bias_gradients, current_learning_rate));
			layer.bias_momentums.values = bias_updates.values.clone();
		}
		// Vanilla SGD no momentum
		else {
			weight_updates = Tensor.multiply(layer.weight_gradients, -1f * current_learning_rate);
			bias_updates = Tensor.multiply(layer.bias_gradients, -1f * current_learning_rate);
		}
		
		// Update weights and biases
		layer.weights = Tensor.add(layer.weights, weight_updates);
		layer.biases = Tensor.add(layer.biases, bias_updates);
	}
	
	// Call after parameter updates
	public void postUpdateParams() {
		iterations += 1;
	}
}
