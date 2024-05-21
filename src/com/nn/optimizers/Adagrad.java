package com.nn.optimizers;

import com.nn.dtypes.Tensor;
import com.nn.layers.*;

public class Adagrad extends Optimizer {
	
	public float decay;
	public float epsilon;
	public int iterations;

	public Adagrad(float learning_rate, float decay, float epsilon) {
		this.learning_rate = learning_rate;
		this.decay = decay;
		this.epsilon = epsilon;
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
		// Initialize cache to 0 if null
		if (layer.weight_cache == null) {
			layer.weight_cache = Tensor.zeros(layer.weights.values.length, layer.weights.values[0].length);
			layer.bias_cache = Tensor.zeros(layer.biases.values.length, layer.biases.values[0].length);
		}
		
		// Update cache with squared gradients
		layer.weight_cache = Tensor.add(layer.weight_cache, Tensor.power(layer.weight_gradients, 2));
		layer.bias_cache = Tensor.add(layer.bias_cache, Tensor.power(layer.bias_gradients, 2));
		
		// Update weights and biases with cache
		layer.weights = Tensor.add(layer.weights, Tensor.divide(Tensor.multiply(layer.weight_gradients, -1f * current_learning_rate), Tensor.add(Tensor.power(layer.weight_cache, 0.5f), epsilon)));
		layer.biases = Tensor.add(layer.biases, Tensor.divide(Tensor.multiply(layer.bias_gradients, -1f * current_learning_rate), Tensor.add(Tensor.power(layer.bias_cache, 0.5f), epsilon)));
	}
	
	// Call after parameter updates
	public void postUpdateParams() {
		iterations += 1;
	}
}
