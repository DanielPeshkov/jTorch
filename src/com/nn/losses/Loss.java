package com.nn.losses;

import com.nn.dtypes.Tensor;

public abstract class Loss {
	float accumulatedSum;
	float accumulatedCount;
	
	public abstract Tensor forward(Tensor predictions, Tensor targets);
	public abstract Tensor backward(Tensor predictions, Tensor targets);
	
	public float calculate(Tensor predictions, Tensor targets) {
		// Calculate sample losses
		Tensor sampleLosses = forward(predictions, targets);
		
		// Calculate mean loss
		float meanLoss = Tensor.mean(sampleLosses);
		
		// Add accumulated sum of losses and sample count
		accumulatedSum += Tensor.sum(sampleLosses, -1).values[0][0];
		accumulatedCount += sampleLosses.values.length;
		
		// Return mean loss
		return meanLoss;
	}
	
	public float calculateAccumulated() {
		return accumulatedSum / accumulatedCount;
	}
	
	// Reset variables for accumulated loss
	public void newPass() {
		accumulatedSum = 0f;
		accumulatedCount = 0f;
	}
}
