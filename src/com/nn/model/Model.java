package com.nn.model;

import java.util.ArrayList;
import java.util.Arrays;
import com.nn.dtypes.Tensor;
import com.nn.layers.*;
import com.nn.activations.*;
import com.nn.losses.*;
import com.nn.metrics.*;
import com.nn.optimizers.*;

public class Model {
	public ArrayList<Layer> layers;
	public Loss loss;
	public Optimizer optimizer;
	public ArrayList<String> metrics;
	public float[] metricResults;
	public float[] metricAccumulated;
	public int metricCount;
	public Layer inputLayer;
	public Layer outputLayer;

	public Model() {
		// Create input layer
		inputLayer = new InputLayer();
	}
	
	// Add layers to the model
	public void add(Layer layer) {
		// Index of previous layer
		int i = layers.size() - 1;
		// Add layer to list
		layers.add(layer);
		// Connect this layer to previous layer
		if (i != -1) {
			layers.get(i).next = layer;
			layer.prev = layers.get(i);
		}
		// Connect first layer to input
		else {
			inputLayer.next = layer;
			layer.prev = inputLayer;
		}
	}
	
	// Set loss and optimizer
	public void set(Loss loss, Optimizer optimizer, ArrayList<String> metrics) {
		this.loss = loss;
		this.optimizer = optimizer;
		this.metrics = metrics;
		
		// Save last layer as output layer
		outputLayer = layers.get(layers.size() - 1);
	}
	
	// Training function
	public void train(Tensor X, Tensor y, Tensor valX, Tensor valY, int epochs, int batchSize, int loggingInterval) {
		
		// Calculate train steps per epoch
		int trainSteps = X.values.length / batchSize;
		// Add extra step if batch size doesn't evenly divide data size
		if (trainSteps * batchSize < X.values.length) {
			trainSteps += 1;
		}
		
		// Main training loop
		for (int epoch = 1; epoch <= epochs; epoch++) {
			// Print epoch number
			System.out.println("Epoch " + String.valueOf(epoch));
			loss.newPass();
			metricAccumulated = new float[metrics.size()];
			metricCount = 0;
			
			for (int step = 0; step < trainSteps; step++) {
				// Slice a batch
				Tensor batchX = new Tensor(batchSize, X.values[0].length);
				Tensor batchY = new Tensor(batchSize, y.values[0].length);
				for (int r = 0; r < batchSize; r++) {
					batchX.values[r] = X.values[step*batchSize + r];
					batchY.values[r] = y.values[step*batchSize + r];
				}
				// Forward pass
				Tensor output = forward(batchX);
				
				// Calculate loss
				float stepLoss = this.loss.calculate(output, batchY);
				// Calculate metrics
				metricResults = calculateMetrics(output, batchY);
				
				// Backward pass
				backward(output, batchY);
				
				// Update parameters
				optimizer.preUpdateParams();
				for (Layer layer : layers) {
					optimizer.updateParams(layer);
				}
				optimizer.postUpdateParams();
				
				// Print a summary
				if (step % loggingInterval == 0 || step == trainSteps - 1) {
					System.out.println(
							"Step: " + step + 
							"\tLoss: " + stepLoss + 
							"\tLearning Rate: " + optimizer.current_learning_rate + 
							"\tMetrics: " + Arrays.toString(metricResults)
							);
				}
			}
			
			// Get epoch loss and metrics
			float epochLoss = loss.calculateAccumulated();
			float[] epochMetrics = metricAccumulated.clone();
			for (int i = 0; i < epochMetrics.length; i++) {
				epochMetrics[i] /= metricCount;
			}
			
			// Print epoch statistics
			System.out.println(
					"Training:: " + 
					"Loss: " + epochLoss + 
					"\tLearning Rate: " + optimizer.current_learning_rate + 
					"\tMetrics: " + Arrays.toString(epochMetrics)
					);
			// Validation check
			evaluate(valX, valY, batchSize);
		}
	}
	
	// Evaluate model on validation data
	public void evaluate(Tensor X, Tensor y, int batchSize) {
		// Reset accumulated values in loss and metrics
		loss.newPass();
		metricAccumulated = new float[metrics.size()];
		metricCount = 0;
		
		// Calculate validation steps
		int valSteps = X.values.length / batchSize;
		// Add extra step if batch size doesn't evenly divide data size
		if (valSteps * batchSize < X.values.length) {
			valSteps += 1;
		}
		
		// Iterate over steps
		for (int step = 0; step < valSteps; step++) {
			// Slice a batch
			Tensor batchX = new Tensor(batchSize, X.values[0].length);
			Tensor batchY = new Tensor(batchSize, y.values[0].length);
			for (int r = 0; r < batchSize; r++) {
				batchX.values[r] = X.values[step*batchSize + r];
				batchY.values[r] = y.values[step*batchSize + r];
			}
			// Forward pass
			Tensor output = forward(batchX);
			
			// Calculate loss
			this.loss.calculate(output, batchY);
			// Calculate metrics
			calculateMetrics(output, batchY);
		}
		// Get epoch loss and metrics
		float valLoss = loss.calculateAccumulated();
		float[] valMetrics = metricAccumulated.clone();
		for (int i = 0; i < valMetrics.length; i++) {
			valMetrics[i] /= metricCount;
		}
		// Print a summary
		System.out.println(
				"Validation:: " + 
				"\tLoss: " + valLoss + 
				"\tMetrics: " + Arrays.toString(valMetrics)
				);
	}
	
	// Predict on samples
	public Tensor predict(Tensor X, int batchSize) {
		// Calculate validation steps
		int steps = X.values.length / batchSize;
		// Add extra step if batch size doesn't evenly divide data size
		if (steps * batchSize < X.values.length) {
			steps += 1;
		}
		
		// List of batch results
		ArrayList<Tensor> batchResults = new ArrayList<Tensor>();
		
		// Iterate over steps
		for (int step = 0; step < steps; step++) {
			// Slice a batch
			Tensor batchX = new Tensor(batchSize, X.values[0].length);
			for (int r = 0; r < batchSize; r++) {
				batchX.values[r] = X.values[step*batchSize + r];
			}
			// Forward pass
			batchResults.add(forward(batchX));
		}
		// Combine batches into output Tensor
		Tensor output = new Tensor(X.values.length, batchResults.get(0).values[0].length);
		for (int step = 0; step < steps; step++) {
			for (int r = 0; r < batchSize; r++) {
				output.values[step*batchSize + r] = batchResults.get(step).values[r];
			}
		}
		return output;
	}
	
	// Forward pass
	public Tensor forward(Tensor X) {
		// Pass data through input layer
		inputLayer.forward(X);
		
		// Call forward method of each layer
		for (Layer layer : layers) {
			layer.forward(layer.prev.outputs);
		}
		return outputLayer.outputs;
	}
	
	// Backward pass
	public void backward(Tensor predictions, Tensor targets) {
		
		// Simplified backprop for softmax classifier
		if (outputLayer instanceof Softmax && loss instanceof CategoricalCrossentropy) {
			// Calculate gradient for combined activation/loss and save to softmax layer
			outputLayer.input_gradients.values = SoftmaxActivationCategoricalCrossentropyLoss.backward(predictions, targets).values;
			
			// Iterate over the rest of the layers
			for (int i = layers.size()-2; i >= 0; i--) {
				layers.get(i).backward(layers.get(i+1).input_gradients);
			}
			return;
		}
		// Pass gradients through output layer
		outputLayer.backward(loss.backward(predictions, targets));
		
		// Iterate over the rest of the layers
		for (int i = layers.size()-2; i >= 0; i--) {
			layers.get(i).backward(layers.get(i+1).input_gradients);
		}
	}
	
	// Calculate metrics
	public float[] calculateMetrics(Tensor predictions, Tensor targets) {
		metricResults = new float[metrics.size()];
		// Iterate over each metric
		for (int i = 0; i < metrics.size(); i++) {
			// Calculate accuracy
			if (metrics.get(i) == "Accuracy") {
				float acc = Accuracy.calculate(predictions, targets);
				metricResults[i] = acc;
				metricAccumulated[i] += acc;
			}
		}
		metricCount += 1;
		return metricResults;
	}
}
