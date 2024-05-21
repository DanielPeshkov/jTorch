package com.nn.activations;

import com.nn.dtypes.Tensor;
import com.nn.layers.Layer;

public class Sigmoid extends Layer {

	public Sigmoid() {
		// TODO Auto-generated constructor stub
	}
	
	public void forward(Tensor inputs) {
		outputs = Tensor.divide(1f, Tensor.add(Tensor.exp(Tensor.multiply(inputs, -1f)), 1f));
	}
	
	public void backward(Tensor gradients) {
		input_gradients = Tensor.multiply(Tensor.multiply(gradients, Tensor.add(Tensor.multiply(outputs, -1f), 1f)), outputs);
	}
}
