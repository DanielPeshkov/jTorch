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
public class Spiral {
	
	public static List<Tensor> createData(int samples, int classes) {
		Random generator = new Random();
		Tensor X = Tensor.zeros(samples*classes, 2);
		Tensor y = Tensor.zeros(samples*classes, 1);
		
		for (int classNum = 0; classNum < classes; classNum++) {
			int startIndex = samples * classNum;
			int endIndex = samples * (classNum + 1);
			
			for (int i = startIndex; i < endIndex; i++) {
				float r = generator.nextFloat();
				float t = 4 * classNum + 4 * (i - startIndex) / (float) samples + generator.nextFloat() * 0.2f;
				X.values[i][0] = r * (float) Math.sin(t * 2.5f);
				X.values[i][1] = r * (float) Math.cos(t * 2.5f);
				y.values[i][0] = classNum;
			}
		}
		
		return Arrays.asList(X, y);
	}
}
