package com.nn.optimizers;

import com.nn.layers.Layer;

public abstract class Optimizer {
	public float learning_rate;
	public float current_learning_rate;

	public abstract void preUpdateParams();
	public abstract void updateParams(Layer layer);
	public abstract void postUpdateParams();
}
