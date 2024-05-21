package com.nn;

import com.nn.dtypes.Tensor;
import com.nn.layers.Dense;
import com.nn.activations.ReLU;
import com.nn.activations.Sigmoid;
import com.nn.activations.Softmax;
import java.util.Arrays;


public class Main {
	
	public void testDenseShape() {
		// 2 input features, 3 nodes in layer
		int input_features = 4;
		int nodes = 3;
		
		float[][] input_values = new float[][] {
			{1f, 2f, 3f, 2.5f}, 
			{1f, 2f, 3f, 2.5f}, 
			{1f, 2f, 3f, 2.5f}, 
		};
		float[][] target_values = new float[][] {
			{0f, 0f, 0f}, 
			{0f, 0f, 0f}, 
			{0f, 0f, 0f}, 
		};
		Tensor input = Tensor.fromFloatArray(input_values);
		Tensor target = Tensor.fromFloatArray(target_values);
		Tensor loss = new Tensor(3, 3);
		
		Dense dense = new Dense(input_features, nodes);
		
		System.out.println("Weights");
		System.out.println(Arrays.deepToString(dense.weights.values).replace("], ", "]\n"));		
		
		System.out.println("\nBiases");
		System.out.println(Arrays.deepToString(dense.biases.values).replace("], ", "]\n"));
		
		System.out.println("\nInput Values");
		System.out.println(Arrays.deepToString(input.values).replace("], ", "]\n"));
		
		dense.forward(input);
		
		System.out.println("\nLayer Output");
		System.out.println(Arrays.deepToString(dense.outputs.values).replace("], ", "]\n"));
	}
	
	public void testDenseTrainLoop() {
		// 2 input features, 3 nodes in layer
		int input_features = 4;
		int nodes = 3;
		
		float[][] input_values = new float[][] {
			{1f, 2f, 3f, 2.5f}, 
			{1f, 2f, 3f, 2.5f}, 
			{1f, 2f, 3f, 2.5f}, 
		};
		float[][] target_values = new float[][] {
			{0f, 0f, 0f}, 
			{0f, 0f, 0f}, 
			{0f, 0f, 0f}, 
		};
		Tensor input = Tensor.fromFloatArray(input_values);
		Tensor target = Tensor.fromFloatArray(target_values);
		Tensor loss = new Tensor(3, 3);
		
		Dense dense = new Dense(input_features, nodes);
		
		dense.forward(input);
		loss = Tensor.multiply(Tensor.subtract(target, dense.outputs), .0001f);
		
		dense.backward(loss);
		dense.weights = Tensor.add(dense.weights,  dense.weight_gradients);
		
		System.out.println("\nWeight Gradients");
		System.out.println(Arrays.deepToString(dense.weight_gradients.values).replace("], ", "]\n"));
		
		System.out.println("\nUpdated Weights");
		System.out.println(Arrays.deepToString(dense.weights.values).replace("], ", "]\n"));	
		
		System.out.println("\nUpdated Output");
		dense.forward(input);
		System.out.println(Arrays.deepToString(dense.outputs.values).replace("], ", "]\n"));
		
		
		for (int i = 0; i < 1000; i++) {
			dense.forward(input);
			loss = Tensor.multiply(Tensor.subtract(target, dense.outputs), .0001f);
			dense.backward(loss);
			dense.weights = Tensor.add(dense.weights,  dense.weight_gradients);
			dense.biases = Tensor.add(dense.biases,  dense.bias_gradients);
			
			//System.out.println("\nUpdated Output");
			//System.out.println(Arrays.deepToString(dense.outputs.values).replace("], ", "]\n"));
		}
		System.out.println("\nUpdated Output");
		System.out.println(Arrays.deepToString(dense.outputs.values).replace("], ", "]\n"));
	}

	public static void main(String[] args) {
		
		// 2 input features, 3 nodes in layer
		int input_features = 4;
		int nodes = 3;
		/*
		float[][] input_values = new float[][] {
			{1f, 2f, 3f, 2.5f}, 
			{1f, 2f, 3f, 2.5f}, 
			{1f, 2f, 3f, 2.5f}, 
		};
		float[][] target_values = new float[][] {
			{0f, 0f, 0f}, 
			{0f, 0f, 0f}, 
			{0f, 0f, 0f}, 
		};
		Tensor input = Tensor.fromFloatArray(input_values);
		Tensor target = Tensor.fromFloatArray(target_values);
		Tensor loss = new Tensor(3, 3);*/
		
		Tensor input = Tensor.random(2, 4);
		
		Dense dense = new Dense(input_features, nodes);
		// ReLU relu = new ReLU();
		// Softmax soft = new Softmax();
		Sigmoid sig = new Sigmoid();
		
		
		dense.forward(input);
		// relu.forward(dense.outputs);
		// soft.forward(dense.outputs);
		sig.forward(dense.outputs);
		
		System.out.println("\nDense Outputs\n" + Arrays.deepToString(dense.outputs.values).replace("], ", "]\n"));
		
		System.out.println("\nSigmoid Outputs\n" + Arrays.deepToString(sig.outputs.values).replace("], ", "]\n"));
		
		sig.backward(sig.outputs);
		System.out.println("\nSigmoid Input Gradients\n" + Arrays.deepToString(sig.input_gradients.values).replace("], ", "]\n"));
		
		dense.backward(sig.input_gradients);
		System.out.println("\nDense Input Gradients\n" + Arrays.deepToString(dense.input_gradients.values).replace("], ", "]\n"));
	}

}
