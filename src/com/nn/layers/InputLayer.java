package com.nn.layers;

import com.nn.dtypes.Tensor;

public class InputLayer extends Layer {

	public InputLayer() {
		// TODO Auto-generated constructor stub
	}

	public  void forward(Tensor inputs) {
		this.inputs = inputs;
		this.outputs = inputs;
	}
	
	public  void backward(Tensor gradients) {
		
	}
}
