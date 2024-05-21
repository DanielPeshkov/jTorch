package com.nn.layers;

import com.nn.dtypes.Tensor;

public abstract class Layer {
	public Tensor inputs;
	public Tensor outputs;
	public Tensor weights;
	public Tensor biases;
	public Tensor weight_gradients;
	public Tensor bias_gradients;
	public Tensor input_gradients;
	
	// Optimizer specific attributes
	public Tensor weight_momentums;
	public Tensor bias_momentums;
	public Tensor weight_cache;
	public Tensor bias_cache;
	
	// Previous and Next layer pointers
	public Layer prev;
	public Layer next;
	
	public abstract void forward(Tensor inputs);
	public abstract void backward(Tensor gradients);
}