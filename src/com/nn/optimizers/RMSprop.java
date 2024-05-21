package com.nn.optimizers;

import com.nn.dtypes.Tensor;
import com.nn.layers.*;

public class RMSprop extends Optimizer {
	
	public float decay;
	public float epsilon;
	public float rho;
	public int iterations;

	public RMSprop(float learning_rate, float decay, float epsilon, float rho) {
		this.learning_rate = learning_rate;
		this.current_learning_rate = learning_rate;
		this.decay = decay;
		this.epsilon = epsilon;
		this.rho = rho;
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
		layer.weight_cache = Tensor.add(Tensor.multiply(layer.weight_cache, rho), Tensor.multiply(Tensor.power(layer.weight_gradients, 2), 1f-rho));
		layer.bias_cache = Tensor.add(Tensor.multiply(layer.bias_cache, rho), Tensor.multiply(Tensor.power(layer.bias_gradients, 2), 1f-rho));
		
		// Update weights and biases with cache
		layer.weights = Tensor.add(layer.weights, Tensor.divide(Tensor.multiply(layer.weight_gradients, -1f * current_learning_rate), Tensor.add(Tensor.power(layer.weight_cache, 0.5f), epsilon)));
		layer.biases = Tensor.add(layer.biases, Tensor.divide(Tensor.multiply(layer.bias_gradients, -1f * current_learning_rate), Tensor.add(Tensor.power(layer.bias_cache, 0.5f), epsilon)));
	}
	
	// Call after parameter updates
	public void postUpdateParams() {
		iterations += 1;
	}
}
