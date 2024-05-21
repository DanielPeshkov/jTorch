package com.nn.optimizers;

import com.nn.dtypes.Tensor;
import com.nn.layers.*;

public class Adam extends Optimizer {
	
	public float decay;
	public float epsilon;
	public float beta_1;
	public float beta_2;
	public int iterations;

	public Adam(float learning_rate, float decay, float epsilon, float beta_1, float beta_2) {
		this.learning_rate = learning_rate;
		this.current_learning_rate = learning_rate;
		this.decay = decay;
		this.epsilon = epsilon;
		this.beta_1 = beta_1;
		this.beta_2 = beta_2;
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
		// Initialize cache and momentum to 0 if null
		if (layer.weight_cache == null) {
			layer.weight_cache = Tensor.zeros(layer.weights.values.length, layer.weights.values[0].length);
			layer.bias_cache = Tensor.zeros(layer.biases.values.length, layer.biases.values[0].length);
			layer.weight_momentums = Tensor.zeros(layer.weights.values.length, layer.weights.values[0].length);
			layer.bias_momentums = Tensor.zeros(layer.biases.values.length, layer.biases.values[0].length);
		}
		
		// Update momentum with current gradients
		layer.weight_momentums = Tensor.add(Tensor.multiply(layer.weight_momentums, beta_1), Tensor.multiply(layer.weight_gradients, 1f-beta_1));
		layer.bias_momentums = Tensor.add(Tensor.multiply(layer.bias_momentums, beta_1), Tensor.multiply(layer.bias_gradients, 1f-beta_1));
		
		// Get corrected momentum
		Tensor weight_momentums_corrected = Tensor.divide(layer.weight_momentums, (float) Math.pow(1-beta_1, iterations+1));
		Tensor bias_momentums_corrected = Tensor.divide(layer.bias_momentums, (float) Math.pow(1-beta_1, iterations+1));
		
		// Update cache with squared gradients
		layer.weight_cache = Tensor.add(Tensor.multiply(layer.weight_cache, beta_2), Tensor.multiply(Tensor.power(layer.weight_gradients, 2), 1f-beta_2));
		layer.bias_cache = Tensor.add(Tensor.multiply(layer.bias_cache, beta_2), Tensor.multiply(Tensor.power(layer.bias_gradients, 2), 1f-beta_2));
		
		// Get corrected cache
		Tensor weight_cache_corrected = Tensor.divide(layer.weight_cache, (float) Math.pow(1-beta_2, iterations+1));
		Tensor bias_cache_corrected = Tensor.divide(layer.bias_cache, (float) Math.pow(1-beta_2, iterations+1));
		
		// Update weights and biases
		layer.weights = Tensor.add(layer.weights, Tensor.divide(Tensor.multiply(weight_momentums_corrected, -1f*current_learning_rate), Tensor.add(Tensor.power(weight_cache_corrected, 0.5f), epsilon)));
		layer.biases = Tensor.add(layer.biases, Tensor.divide(Tensor.multiply(bias_momentums_corrected, -1f*current_learning_rate), Tensor.add(Tensor.power(bias_cache_corrected, 0.5f), epsilon)));
	}
	
	// Call after parameter updates
	public void postUpdateParams() {
		iterations += 1;
	}
}
