package zeroPerception;

import java.util.ArrayList;
import java.util.List;

/**
 * @author z3r0r4
 * @version 2.0
 */

public class Perceptron {
	private int No_L;// number of Layers
	private int t = 0;//number of iterations
	private double η = 0.1; //learning rate
	private Matrix[] Layers, Biases, Weights;
	private Matrix[] δ, δCost_δWeights, δCost_δBiases;//previously δCost_δLayers = δ
	private Matrix Target;

	private List<Matrix[]> C = new ArrayList<Matrix[]>(); //for batches Δ C={δCost_δWeights, δCost_δBiases}
	private Matrix[] MeanδCost_δWeights, MeanδCost_δBiases;

	public Perceptron(int... NumberOfNodes) { //NumberOfNodes[Layer] Anzahl der nodes im layer 'Layer' 
		System.out.println("++++++++INITIALISING+++++++++");
		No_L = NumberOfNodes.length; //NumberOfNodes.length Anzahl der Layerrs

		Layers = new Matrix[No_L];
		Weights = Biases = new Matrix[No_L - 1];
		Weights = new Matrix[No_L - 1];
		δ = new Matrix[No_L];
		δCost_δBiases = new Matrix[No_L - 1];
		δCost_δWeights = new Matrix[No_L - 1];
		MeanδCost_δBiases = new Matrix[No_L - 1];
		MeanδCost_δWeights = new Matrix[No_L - 1];

		for (int i = 0; i < No_L; i++) {
			Layers[i] = new Matrix(NumberOfNodes[i], 1); //filled with 0
			δ[i] = new Matrix(Layers[i].getRows(), 1); // filled with 0
		}

		for (int i = 0; i < No_L - 1; i++) {
			Biases[i] = new Matrix(Layers[i + 1].getRows(), 1, 1);
			Weights[i] = new Matrix(Layers[i + 1].getRows(), Layers[i].getRows(), 1);
			δCost_δBiases[i] = new Matrix(Biases[i].getRows(), 1);
			δCost_δWeights[i] = new Matrix(Layers[i + 1].getRows(), Layers[i].getRows());
			MeanδCost_δBiases[i] = new Matrix(Biases[i].getRows(), 1);
			MeanδCost_δWeights[i] = new Matrix(Layers[i + 1].getRows(), Layers[i].getRows());
		}
	}

	public int getNumber_of_iterations() {
		return t;
	}

	public void Train(int Iterations, double[][] input, double[][] Target) {
		for (int it = 0; it < Iterations; it++) {
			forwardProp(input);
			backProp(Target);
			Apply(δCost_δWeights, δCost_δBiases);
			t++;
			//System.out.println(cost(Target));
			//learningrate();
		}
	}

	public void Train(int Iterations, double[][] input, double[][] Target, int No) {
		for (int it = 0; it < Iterations; it++) {
			C = new ArrayList<Matrix[]>();
			for (int i = 0; i < No; i++) {
				forwardProp(input);
				backProp(Target);
				C.add(δCost_δWeights);
				C.add(δCost_δBiases);
				t++;
			}
			for (int i = 0; 2 * i + 1 < C.size(); i++) {
				for (int n = 0; n < No_L - 1; n++) {
					MeanδCost_δWeights[n].add(C.get(2 * i)[n]);
					MeanδCost_δWeights[n].scalarmult(1 / (No - 1));
					MeanδCost_δBiases[n].add(C.get(2 * i + 1)[n]);
					MeanδCost_δBiases[n].scalarmult(1 / (No - 1));
				}
			}

			Apply(MeanδCost_δWeights, MeanδCost_δBiases);

			System.out.println("COST: " + cost(Target));
			//learningrate();
		}
	}

	public void guess(double[][] input) {
		forwardProp(input);
		System.out.println(cost(Target.getData()));
	}

	public void infoL() {
		for (int i = 0; i < No_L - 1; i++) {
			System.out.println("\nLayer " + i);
			System.out.print("a: ");
			Matrix.printM(Layers[i]);
			System.out.print("w: ");
			Matrix.printM(Weights[i]);
			System.out.print("b: ");
			Matrix.printM(Biases[i]);
		}
		System.out.print("a: ");
		Matrix.printM(Layers[No_L - 1]);
		System.out.print("y: ");
		Matrix.printM(Target);
	}

	public void infoT() {
		System.out.print("\nInput: ");
		Matrix.printM(Layers[0]);
		System.out.print("Guess: ");
		Matrix.printM(Layers[No_L - 1]);
		System.out.print("Target: ");
		Matrix.printM(Target);
		cost(Target.getData());
	}

	private void forwardProp(double[][] input) {
		//System.out.println("\n++++++++FORWARD-PROPAGATION++++++++++" + t);

		Matrix Layer = Matrix.fromArray(input);
		Layers[0] = Layer;
		int i = 1;
		for (Matrix weight : Weights) {
			Layer = sigmoid(Matrix.add(Matrix.prod(weight, Layer), Biases[i - 1]));
			Layers[i] = Layer;
			i++;
		}
	}

	private void backProp(double[][] y) {
		//System.out.println("\n++++++++BACKWARD-PROPAGATION+++++++++" + t);

		Target = Matrix.fromArray(y);
		for (int n = No_L - 1; n >= 1; n--) {
			if (n == No_L - 1)
				δ[n] = Matrix.scalarmult(2, Matrix.add(Layers[n], Matrix.negate(Target)));
			else if (n != No_L - 1)
				δ[n] = Matrix.mult(
						Matrix.prod(Matrix.T(Weights[n]), δ[n + 1]),
						Matrix.mult(Layers[n], Matrix.negate(Layers[n]).add(1))); //sigmoid_1(z[n])
			δCost_δWeights[n - 1] = Matrix.prod(δ[n], Matrix.T(Layers[n - 1]));
			δCost_δBiases[n - 1] = δ[n];
		}
	}

	private double cost(double[][] Target) {
		double a = 0;
		for (int i = 0; i < Layers[No_L - 1].getRows(); i++)
			a += Math.pow((Layers[No_L - 1].getData(i, 0) - Target[i][0]), 2);
		return a;
	}

	private void Apply(Matrix[] Δw, Matrix[] Δb) {
		for (int n = No_L - 2; n > 0; n--) {
			Biases[n].add(Matrix.scalarmult(η, Matrix.negate(Δb[n])));
			Weights[n].add(Matrix.scalarmult(η, Matrix.negate(Δw[n])));
		}
	}

	private void learningrate() {
		η = 0 / Math.sqrt(t);
	}


	//SIGMOID THINGS
	private Matrix sigmoid(Matrix A) {
		Matrix B = new Matrix(A.getRows(), A.getColumns());
		for (int i = 0; i < A.getRows(); i++)
			for (int j = 0; j < A.getColumns(); j++) {
				B.setData(sigmoid(A.getData()[i][j]), i, j);
			}
		return B;
	}

	private double sigmoid(double a) {
		return 1 / (1 + Math.exp(-a));
	}

}
