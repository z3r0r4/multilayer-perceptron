package zeroPerception;

import java.util.ArrayList;

public class Main {

	static String Path = ".\\doc\\mnist_train.csv";
	static ArrayList<double[]> input = ImageData.input(Path);
	static ArrayList<double[]> target = ImageData.target();

	public static void main(String[] args) {

		Perceptron Percepti = new Perceptron(784, 200, 10);
		Percepti.Benchmark(600000, input, target);

	}
}
