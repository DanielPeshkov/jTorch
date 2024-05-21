package com.nn.losses;

import com.nn.dtypes.Tensor;

public class BinaryCrossentropy extends Loss{

	public BinaryCrossentropy() {
		// TODO Auto-generated constructor stub
	}
	
	public Tensor forward(Tensor predictions, Tensor targets) {
		// Clip data to prevent divide by 0
		// Clip both sides to prevent bias towards 1
		Tensor clipped = Tensor.clip(predictions, 1E-7f, 1f-1E-7f);
		
		// Calculate sample-wise loss
		Tensor output = Tensor.multiply(Tensor.add(Tensor.multiply(targets, Tensor.log(clipped)), Tensor.multiply(Tensor.add(Tensor.multiply(targets, -1f), 1f), Tensor.log(Tensor.add(Tensor.multiply(clipped, -1f), 1f)))), -1f);
		output = Tensor.multiply(Tensor.sum(output, 1), 1f/output.values[0].length);
		
		return output;
	}
	
	public Tensor backward(Tensor predictions, Tensor targets) {
		// Number of samples in prediction (batch size)
		int samples = predictions.values.length;
		// Number of outputs per sample
		int outputs = predictions.values[0].length;
		
		// Clip data to prevent divide by 0
		// Clip both sides to prevent bias towards 1
		Tensor clipped = Tensor.clip(predictions, 1E-7f, 1f-1E-7f);
		
		// Calculate gradient
		Tensor input_gradients = Tensor.multiply(Tensor.multiply(Tensor.subtract(Tensor.divide(targets, clipped), Tensor.divide(Tensor.add(Tensor.multiply(targets, -1f), 1f), Tensor.add(Tensor.multiply(clipped, -1f), 1f))), -1f), 1f/outputs);
		// Normalize gradient
		input_gradients = Tensor.multiply(input_gradients, 1f/samples);
		
		return input_gradients;
	}
}
