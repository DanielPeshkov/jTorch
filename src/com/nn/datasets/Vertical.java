package com.nn.datasets;

import com.nn.dtypes.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/*
 * Java adaptation of the following source of Python code: 
 * Copyright (c) 2015 Andrej Karpathy
 * License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
 * Source: https://cs231n.github.io/neural-networks-case-study/
 */
public class Vertical {
	
	public static List<Tensor> createData(int samples, int classes) {
		Random generator = new Random();
		Tensor X = Tensor.zeros(samples*classes, 2);
		Tensor y = Tensor.zeros(samples*classes, 1);
		
		for (int classNum = 0; classNum < classes; classNum++) {
			int startIndex = samples * classNum;
			int endIndex = samples * (classNum + 1);
			
			for (int i = startIndex; i < endIndex; i++) {
				float x1 = generator.nextFloat() * 0.1f + (float) classNum / 3f;
				float x2 = generator.nextFloat() * 0.1f + 0.5f;
				X.values[i][0] = x1;
				X.values[i][1] = x2;
				y.values[i][0] = classNum;
			}
		}
		
		return Arrays.asList(X, y);
	}
}
