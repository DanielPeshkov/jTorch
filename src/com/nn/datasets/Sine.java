package com.nn.datasets;

import com.nn.dtypes.Tensor;

import java.util.Arrays;
import java.util.List;

/*
 * Java adaptation of the following source of Python code: 
 * Copyright (c) 2020 Daniel Kukiela and Harrison Kinsley
 * License: https://github.com/Sentdex/nnfs/blob/master/LICENSE
 * Source: https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/sine.py
 */

public class Sine {
	
	public static List<Tensor> createData(int samples) {
		Tensor X = Tensor.zeros(samples, 2);
		Tensor y = Tensor.zeros(samples, 1);
		
		for (int i = 0; i < samples; i++) {
			float x = (float) i / samples;
			X.values[i][0] = x;
			y.values[i][0] = (float) Math.sin(2 * Math.PI * x);
		}
		
		return Arrays.asList(X, y);
	}
}
