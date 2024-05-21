package com.nn.dtypes;

import java.util.Random;

public class Tensor {
	
	// 2D tensor of floats that holds data in Tensor
	public float[][] values;
	
	// Create an empty Tensor of size (rows, cols)
	public Tensor(int rows, int cols) {
		values = new float[rows][cols];
	}
	
	// Fill Tensor with zeros
	public static Tensor zeros(int rows, int cols) {
		Tensor tensor = new Tensor(rows, cols);
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				tensor.values[i][j] = 0f;
			}
		}
		return tensor;
	}
	
	// Fill Tensor with zeros
	public static Tensor ones(int rows, int cols) {
		Tensor tensor = new Tensor(rows, cols);
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				tensor.values[i][j] = 1f;
			}
		}
		return tensor;
	}
	
	// Return Tensor with random values between -1 and 1
	public static Tensor random(int rows, int cols) {
		// Random number generator
		Random r = new Random();
		Tensor tensor = new Tensor(rows, cols);
		
		// Iterate to fill Tensor
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				tensor.values[i][j] = r.nextFloat(-1f, 1f);
			}
		}
		return tensor;
	}
	
	// Return Tensor with random values between min and max
	public static Tensor random(int rows, int cols, float min, float max) {
		// Random number generator
		Random r = new Random();
		Tensor tensor = new Tensor(rows, cols);
		
		// Iterate to fill Tensor
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				tensor.values[i][j] = r.nextFloat(min, max);
			}
		}
		return tensor;
	}
	
	// Create a Tensor from tensor of floats
		public static Tensor fromFloatArray(float[][] values) {
			Tensor tensor = new Tensor(values.length, values[0].length);
			
			// Copy values into Tensor
			for (int i = 0; i < values.length; i++) {
				for (int j = 0; j < values[0].length; j++) {
					tensor.values[i][j] = values[i][j];
				}
			}
			return tensor;
		}
	
	// Transpose and return copy of Tensor
	public Tensor T() {
		int rows = values.length;
		int cols = values[0].length;
		
		Tensor tensor = new Tensor(cols, rows);
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				tensor.values[j][i] = values[i][j];
			}
		}
		
		return tensor;
	}
	
	// Calculate the dot product of 2 Tensors
	public static Tensor dot(Tensor t, Tensor t2) {
		// Output tensor shape
		int rows = t.values.length;
		int cols = t2.values[0].length;
		int vecLen = t2.values.length;
		
		// Create output tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell in output
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				// Multiply and sum each vector
				for (int i = 0; i < vecLen; i++) {
					output.values[r][c] += t.values[r][i] * t2.values[i][c];
				}
			}
		}
		return output;
	}	
	
	// Add 2 Tensors
	public static Tensor add(Tensor t, Tensor t2) {
		// Output Tensor Shape
		int rows = t2.values.length;
		// int cols = t2.values[0].length;
		
		// Check if 1D tensor is being added to 2D tensor
		if (rows == 1) {
			return add2Dand1D(t, t2);
		}
		else {
			return addEqualShape(t, t2);
		}
	}
	
	// Add 2 Tensors of equal shape
	public static Tensor addEqualShape(Tensor t, Tensor t2) {
		// Output Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] + t2.values[r][c];
			}
		}
		return output;
	}
	
	// Add 1D Tensor to Batch of 1D Tensors (2D shape)
	public static Tensor add2Dand1D(Tensor t, Tensor t2) {
		// Output Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] + t2.values[0][c];
			}
		}
		return output;
	}
	
	// Add float to each value in Tensor
	public static Tensor add(Tensor t, float num) {
		// Output Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] + num;
			}
		}
		return output;
	}
	
	// Sum Tensor along dim
	public static Tensor sum(Tensor t, int dim) {
		if (dim == 0) {
			return sum0(t);
		}
		else if (dim == 1) {
			return sum1(t);
		}
		else {
			return sumAll(t);
		}
	}
	
	// Sum all values in Tensor
	public static Tensor sumAll(Tensor t) {
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = zeros(1, 1);
	
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[0][0] += t.values[r][c];
			}
		}
		return output;
	}
	
	// Sum 2D Tensor along dim=0
	public static Tensor sum0(Tensor t) {
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = zeros(1, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[0][c] += t.values[r][c];
			}
		}
		return output;
	}
	
	// Sum 2D Tensor along dim=1
	public static Tensor sum1(Tensor t) {
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = zeros(rows, 1);
	
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][0] += t.values[r][c];
			}
		}
		return output;
	}
	
	// Call the correct subtract based on Tensor shape
	public static Tensor subtract(Tensor t, Tensor t2) {
		int rows = t2.values.length;
		int cols = t2.values[0].length;
		
		if (cols == 1 && rows != 1) {
			return subtractColFromTensor(t, t2);
		}
		else if (cols == 1 && rows == 1) {
			return subtractScalar(t, t2);
		}
		else {
			return subtractEqualShape(t, t2);
		}
	}
	
	// Subtract one Tensor from another with equal shape
	public static Tensor subtractEqualShape(Tensor t, Tensor t2) {
		// Output Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] - t2.values[r][c];
			}
		}
		return output;
	}
	
	// Subtract single column Tensor from 2D Tensor
	public static Tensor subtractColFromTensor(Tensor t, Tensor t2) {
		// Output Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] - t2.values[r][0];
			}
		}
		return output;
	}
	
	// Subtract float from each element of Tensor
	public static Tensor subtractScalar(Tensor t, Tensor t2) {
		// Output Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] - t2.values[0][0];
			}
		}
		return output;
	}
	
	// Multiply Tensor by float
	public static Tensor multiply(Tensor t, float scale) {
		// Output Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] * scale;
			}
		}
		return output;
	}
	
	// Multiply 2 Tensors element-wise
	public static Tensor multiply(Tensor t, Tensor t2) {
		// Output shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] * t2.values[r][c];
			}
		}
		return output;
	}
	
	// Divide Tensor
	public static Tensor divide(Tensor t, Tensor t2) {
		int rows = t2.values.length;
		int cols = t2.values[0].length;
		
		// Divide t by column Tensor t2
		return divideByCol(t, t2);
	}
	
	// Divide Tensor by single column Tensor
	public static Tensor divideByCol(Tensor t, Tensor t2) {
		// Output shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] / t2.values[r][0];
			}
		}
		return output;
	}
	
	// Divide Float by each value in Tensor
	public static Tensor divide(float num, Tensor t) {
		// Output shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = num / t.values[r][c];
			}
		}
		return output;
	}
	
	// Divide Float by each value in Tensor
	public static Tensor divide(Tensor t, float num) {
		// Output shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = t.values[r][c] / num;
			}
		}
		return output;
	}
	
	// Divide values in 2 equal-shaped Tensors with checks for 0
	public static Tensor divideWithChecks(Tensor t, Tensor t2) {
		// Output shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (t.values[r][c] == 0) {
					output.values[r][c] = 0f;
				} else if (t2.values[r][c] == 0) {
					output.values[r][c] = Float.MAX_VALUE;
				} else {
					output.values[r][c] = t.values[r][c] / t2.values[r][0];
				}
			}
		}
		return output;
	}
	
	// Mean of all values in Tensor
	public static float mean(Tensor t) {
		// Output shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output value and counter
		float output = 0f;
		int count = 0;
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output += t.values[r][c];
				count += 1;
			}
		}
		output /= count;
		return output;
	}
	
	// Mean of values across dim 1
	public static Tensor mean(Tensor t, int dim) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = Tensor.zeros(rows, 1);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][0] += t.values[r][c];
			}
			output.values[r][0] /= (float) cols;
		}
		return output;
	}
	
	// Convert one-hot encoding to index across batch
	public static Tensor argmax(Tensor t) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, 1);
		
		// Placeholder variable for max value
		float best;
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			best = t.values[r][0];
			output.values[r][0] = 0f;
			for (int c = 0; c < cols; c++) {
				if (t.values[r][c] > best) {
					best = t.values[r][c];
					output.values[r][0] = c;
				}
			}
		}
		return output;
	}
	
	// Return max value in each row
	public static Tensor max(Tensor t) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, 1);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			output.values[r][0] = t.values[r][0];
			for (int c = 0; c < cols; c++) {
				if (t.values[r][c] > output.values[r][0]) {
					output.values[r][0] = t.values[r][c];
				}
			}
		}
		return output;
	}
	
	// Exponentiate each value in Tensor with Euler's number
	public static Tensor exp(Tensor t) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = (float) Math.exp( t.values[r][c]);
			}
		}
		return output;
	}
	
	// Create 2D Tensor with flattened input across diagonal
	public static Tensor diagflat(Tensor t) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows * cols, rows * cols);
		
		int index;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				index = r * (1+c) + c;
				output.values[index][index] = t.values[r][c];
			}
		}
		return output;
	}
	
	// Create a binary mask Tensor with a masking rate
	public static Tensor binaryMask(float rate, int rows, int cols) {
		// Output Tensor
		Tensor mask = ones(rows, cols);
		// Random number generator
		Random gen = new Random();
		
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (gen.nextFloat() < rate) {
					mask.values[r][c] = 0f;
				}
			}
		}
		return mask;
	}
	
	// Clip all values in Tensor between min and max
	public static Tensor clip(Tensor t, float min, float max) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = Math.max(min, t.values[r][c]);
				output.values[r][c] = Math.min(max, output.values[r][c]);
			}
		}
		return output;
	}
	
	// Calculate log of each value in Tensor
	public static Tensor log(Tensor t) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] =(float) Math.log(t.values[r][c]);
			}
		}
		return output;
	}
	
	// Convert sparse values to one-hot
	public static Tensor sparseToOneHot(Tensor t, int categories) {
		// Input Tensor Shape
		int rows = t.values.length;
		
		// Create output Tensor
		Tensor output = Tensor.zeros(rows, categories);
		
		// Select one-hot column for each row
		for (int r = 0; r < rows; r++) {
			output.values[r][(int) t.values[r][0]] = 1f;
		}
		return output;
	}
	
	// Raise values in Tensor to a power
	public static Tensor power(Tensor t, float num) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = (float) Math.pow(t.values[r][c], num);
			}
		}
		return output;
	}
	
	// Absolute values of Tensor
	public static Tensor abs(Tensor t) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = Math.abs(t.values[r][c]);
			}
		}
		return output;
	}
	
	// Creates a Tensor with the sign of each value in Tensor
	public static Tensor sign(Tensor t) {
		// Input Tensor Shape
		int rows = t.values.length;
		int cols = t.values[0].length;
		
		// Create output Tensor
		Tensor output = new Tensor(rows, cols);
		
		// Iterate over each cell
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				output.values[r][c] = Math.signum(t.values[r][c]);
			}
		}
		return output;
	}
}








