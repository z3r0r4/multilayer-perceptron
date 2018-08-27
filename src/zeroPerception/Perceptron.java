package zeroPerception;

import java.util.ArrayList;
import java.util.Random;

public class Perceptron {

	private int No_L; // number of Layers
	private int t = 0; //number of iterations
	private double η = 0.02; //learning rate

	private Matrix[] Layers, Biases, Weights;
	private Matrix[] δ, δCost_δWeights, δCost_δBiases;
	private long lStartTime, lEndTime;
	private Random random1 = new Random(0L);
	private Random random2 = new Random(0L);

	public Perceptron(int... NumberOfNodes) {
		System.out.println("+STARTING initialization");

		No_L = NumberOfNodes.length;
		Layers = new Matrix[No_L];
		Weights = Biases = new Matrix[No_L - 1];
		Weights = new Matrix[No_L - 1];
		δ = new Matrix[No_L];
		δCost_δBiases = new Matrix[No_L - 1];
		δCost_δWeights = new Matrix[No_L - 1];

		for (int i = 0; i < No_L; i++) {
			Layers[i] = new Matrix(NumberOfNodes[i], 1); //filled with 0
			δ[i] = new Matrix(Layers[i].getRows(), 1); // filled with 0
		}
		for (int i = 0; i < No_L - 1; i++) {
			Biases[i] = new Matrix(Layers[i + 1].getRows(), 1, -1, 1); //filled with [-1;1]
			Weights[i] = new Matrix(Layers[i + 1].getRows(), Layers[i].getRows(), -1, 1); //filled with [-1;1]
			δCost_δBiases[i] = new Matrix(Biases[i].getRows(), 1);
			δCost_δWeights[i] = new Matrix(Layers[i + 1].getRows(), Layers[i].getRows());
		}
		System.out.println("-FINISHED initialization");
	}

	public void Train(int Iterations, ArrayList<double[]> input, ArrayList<double[]> target) { //goes through all of the dataset randomly
		System.out.println("+STARTING Training");
		for (int it = 0; it < Iterations; it++) {
			if (t % 1000 == 0)
				System.out.println(t);
			forwardProp(input.get(random1.nextInt(input.size())));//forwardProp(input.get(it));
			backProp(target.get(random2.nextInt(target.size())));
			Apply(δCost_δWeights, δCost_δBiases);
			//learningRate();
			t++;
		}
		System.out.println("-FINISHED Training " + t);
	}

	public void Benchmark(int Iterations, ArrayList<double[]> input, ArrayList<double[]> target) { //goes through all of the dataset randomly
		System.out.println("+STARTING Benchmark");
		lStartTime = System.nanoTime();
		Train(Iterations, input, target);
		timer();
		guess(input.get(1), target.get(1));
		guess(input.get(3), target.get(3));
		guess(input.get(5), target.get(5));
		guess(input.get(7), target.get(7));
		guess(input.get(2), target.get(2));
		guess(input.get(0), target.get(0));
		guess(input.get(13), target.get(13));
		guess(input.get(15), target.get(15));
		guess(input.get(17), target.get(17));
		guess(input.get(19), target.get(19));
		test(10000, input, target);
		System.out.println("-FINISHED Benchmark of " + t + " Iterations");
	}

	public void test(int Iterations, ArrayList<double[]> input, ArrayList<double[]> target) {
		double[] Correctness = new double[target.get(0).length];
		int[] Number = new int[target.get(0).length];
		for (int i = 0; i < Iterations; i++) {
			int a = (int) (Math.random() * input.get(0).length);
			double[] guss = guess(input.get(a), target.get(a));
			Number[(int) guss[0]]++;
			if (guss[0] == guss[1])
				Correctness[(int) guss[0]] += 1;
		}
		for (int i = 0; i < target.get(0).length; i++)
			Correctness[i] /= Number[i] * 0.01;
		System.out.println("The average Correctness of " + Iterations + " test's of the Perceptron after " + t
				+ " Iterations is:");
		for (int i = 0; i < Correctness.length; i++)
			System.out.println("Number: " + i + ":  " + Math.round(Correctness[i]) + "%");
	}

	public double[] guess(double[] input, double[] target) {
		forwardProp(input);
		//		System.out.println("Guessed Array:");
		//	Matrix.printM(Layers[No_L - 1]);
		double[] g = Matrix.toArray_flat(Layers[No_L - 1]);
		int indexTarget = 0, index = 0;
		for (int i = 0; i < 10; i++) {
			indexTarget = (target[i] > target[index]) ? i : indexTarget;
			index = (g[i] > g[index]) ? i : index;
		}
		//		System.out.println(
		//				"The guessed Number is maybe, maybe not:\n"
		//						+ index + "\nIt should have been:\n" + indexTarget
		//						+ "\nThe Cost of the guess is:");
		//		System.out.println(cost(target));
		//		System.out.println();
		return new double[] { index, indexTarget, cost(target) };
	}

	private void forwardProp(double[] input) {
		Matrix Layer = Matrix.fromArray(input);
		Layers[0] = Layer;
		int i = 1;
		for (Matrix weight : Weights) {
			Layer = sigmoid(Matrix.add(Matrix.prod(weight, Layer), Biases[i - 1]));
			Layers[i] = Layer;
			i++;
		}
	}

	private void backProp(double[] y) {
		Matrix Target = Matrix.fromArray(y);
		for (int n = No_L - 1; n >= 1; n--) {
			if (n == No_L - 1)
				δ[n] = Matrix.scalarmult(2, Matrix.add(Layers[n], Matrix.negate(Target)));
			else if (n != No_L - 1)
				δ[n] = Matrix.mult(
						Matrix.prod(Matrix.T(Weights[n]), δ[n + 1]),
						Matrix.mult(Layers[n], Matrix.negate(Layers[n]).add(1)));

			δCost_δWeights[n - 1] = Matrix.prod(δ[n], Matrix.T(Layers[n - 1]));
			δCost_δBiases[n - 1] = δ[n];
		}
		if (t % 1000 == 0)
			System.out.println("Cost: " + cost(y));
	}

	private void Apply(Matrix[] Δw, Matrix[] Δb) {

		for (int n = No_L - 2; n > 0; n--) {
			Biases[n].add(Matrix.scalarmult(η, Matrix.negate(Δb[n])));
			Weights[n].add(Matrix.scalarmult(η, Matrix.negate(Δw[n])));
		}
	}

	private double cost(double[] Target) {
		double a = 0;
		for (int i = 0; i < Layers[No_L - 1].getRows(); i++)
			a += Math.pow((Layers[No_L - 1].getData(i, 0) - Target[i]), 2);
		return a;
	}

	private void learningRate() {
		η -=0.00000005;
		//System.out.println(η);
	}

	//SIGMOID THINGS
	private Matrix sigmoid(Matrix A) {
		for (int i = 0; i < A.getRows(); i++)
			for (int j = 0; j < A.getColumns(); j++)
				A.setData(sigmoid(A.getData()[i][j]), i, j);
		return A;
	}

	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	private void timer() {
		lEndTime = System.nanoTime();
		double output, outputmillisec, outputsec, outputminutes, outputhours, outputdays, outputweeks, outputyears;

		output = lEndTime - lStartTime;
		outputmillisec = output / 1E6;
		outputsec = output / 1E9;
		outputminutes = outputsec / 60;
		outputhours = outputminutes / 60;
		outputdays = outputhours / 24;
		outputweeks = outputdays / 7;
		outputyears = outputweeks / 51;
		System.out.println("\nTraining FINISHED");
		System.out.println("Elapsed time in milliseconds: " + outputmillisec);
		System.out.println("Elapsed time in seconds		: " + outputsec);
		System.out.println("Elapsed time in minutes		: " + outputminutes);
		System.out.println("Elapsed time in hours		: " + outputhours);
		System.out.println("Elapsed time in days		: " + outputdays);
		System.out.println("Elapsed time in weeks		: " + outputweeks);
		System.out.println("Elapsed time in years		: " + outputyears);
		System.out.println("Seconds needed for one Iteration: " + outputsec / t + "\n");
	}
}
