package zeroPerception;

import java.util.ArrayList;
import java.util.List;

import zeroMath.Matrix;

/**
 * @author z3r0r4
 * @version 1.0
 */

public class Perceptron {
	public Matrix[] Layers, z, Bias, weights;
	public Matrix[] δCost_δLayers, δCost_δWeights, δCost_δBiases;
	public int No_l;// number of Layers

	public Matrix y;
	public List<Double> C = new ArrayList<Double>();

	public int t = 0;
	public double lamda = 0.01;

	public Perceptron(int... NumberOfNodes) { //NumberOfNodes[Layer] Anzahl der nodes im layer 'Layer' 
		System.out.println("++++++++INITIALISING+++++++++");
		No_l = NumberOfNodes.length; //NumberOfNodes.length Anzahl der Layerrs

		Layers = new Matrix[No_l];
		for (int i = 0; i < No_l; i++) {
			Layers[i] = new Matrix(NumberOfNodes[i], 1, 0); //filled with 0

		}

		//z = new Matrix[No_l - 1];
		Bias = new Matrix[No_l - 1];
		weights = new Matrix[No_l - 1];
		for (int i = 0; i < No_l - 1; i++) {
			//z[i] = new Matrix(Layers[i + 1].getRows(), 1,0,0); // filled with 0
			Bias[i] = new Matrix(Layers[i + 1].getRows(), 1, 1, 10); // filled with random data
			weights[i] = new Matrix(Layers[i + 1].getRows(), Layers[i].getRows(), 1, 10); // filled with random data
		}

		δCost_δLayers = new Matrix[No_l];
		δCost_δBiases = new Matrix[No_l - 1];
		δCost_δWeights = new Matrix[No_l - 1];
		for (int i = 0; i < No_l - 1; i++) {

			δCost_δLayers[i] = new Matrix(Layers[i].getRows(), 1); // filled with 0
			δCost_δBiases[i] = new Matrix(Bias[i].getRows(), 1); // filled with 0
			δCost_δWeights[i] = new Matrix(Layers[i + 1].getRows(), Layers[i].getRows()); // filled with 0
		}
		δCost_δLayers[No_l - 1] = new Matrix(Layers[No_l - 1].getRows(), 1); // filled with 0 // Node Gradient for the last layer

		y = new Matrix(Layers[Layers.length - 1].getRows(), 1);
		for (int i = 0; i < y.getRows(); i++)
			y.setData(10, i, 0); // filled with 0 for now atleast

	}

	public void forwardProp(double[][] input) {//takes the input for the network as input
		System.out.println("\n++++++++FORWARD-PROPAGATION++++++++++" + t);
		t++;
		Layers[0].setData(input);
		Matrix Layer = Matrix.fromArray(input);//happy now kai?
		int i = 1;
		for (Matrix weight : weights) {
			Layer = sigmoid(Matrix.add(Matrix.prod(weight, Layer), Bias[i - 1]));
			Layers[i] = Layer;
			i++;
		}
	}

	public void backProp(double[][] correctAnswer) {
		System.out.println("\nError of Propagation " + t + " is:\n" + C.get(t - 1) + "\n");

		System.out.println("\n++++++++BACKWARD-PROPAGATION+++++++++" + t);
		y = Matrix.fromArray(correctAnswer);
		//Gradient_a computation for the last Layer
		δCost_δLayers[No_l - 1] = Matrix.scalarmult(2, Matrix.add(Layers[No_l - 1], Matrix.negate(y)));
		//Gradient_a computation for all layers except the last
		for (int n = No_l - 2; n > 0; n--) {
			δCost_δLayers[n] = Matrix.prod(Matrix.transpose(weights[n]),
					Matrix.mult(
							Matrix.mult(Layers[n + 1], Matrix.negate(Layers[n + 1]).add(1)), //==sigmoid_1(z[n])
							δCost_δLayers[n + 1]));
		}
		//Gradient_b computation for all layers except the last
		for (int n = No_l - 2; n > 0; n--) {
			δCost_δBiases[n] = Matrix.mult(
					Matrix.mult(Layers[n + 1], Matrix.negate(Layers[n + 1]).add(1)), //==sigmoid_1(z[n].getData(l, 0))
					δCost_δLayers[n + 1]);
			//Gradient_w computation for all layers except the last
			δCost_δWeights[n] = Matrix.prod(
					Matrix.mult(Layers[n + 1], Matrix.negate(Layers[n + 1]).add(1)), //==sigmoid_1(z[n])
					Matrix.transpose(Matrix.mult(Layers[n], δCost_δLayers[n])));
		}
	}

	public void forwardInfo() {
		for (int i = 0; i < No_l - 1; i++)
			System.out.printf("\n%d. Layer's | Nodes: %d & Z's: %d & biases: %d & weights %dx%d  ", i,
					Layers[i].getRows(),
					z[i].getRows(), Bias[i].getRows(), weights[i].getRows(), weights[i].getColumns());
		// info about last layer
		System.out.printf("%d. Layer's | Nodes: %d & y's: %d  \n", No_l - 1, Layers[No_l - 1].getRows(), y.getRows());
	}

	public void backwardInfo() {
		for (int i = 0; i < No_l - 1; i++)
			System.out.printf("\n%d. Layer's | Node Gradients: %d & Bias Gradients: %d & Weight Gradients %dx%d  \n", i,
					δCost_δLayers[i].getRows(), δCost_δBiases[i].getRows(), δCost_δWeights[i].getRows(),
					δCost_δWeights[i].getColumns());
		//no info about last layer
	}

	public void APPLY() {
		for (int n = No_l - 2; n >= 0; n--) {
			Bias[n].add(Matrix.scalarmult(lamda, Matrix.negate(δCost_δBiases[n])));
			weights[n].add(Matrix.scalarmult(lamda, Matrix.negate(δCost_δWeights[n])));
		}
	}

	public void cost() { //total error of one propagation
		C.add(0.0);
		for (int i = 0; i < Layers[No_l - 1].getRows(); i++)
			C.add(t - 1, C.get(t - 1) + Math.pow((Layers[No_l - 1].getData(i, 0) - y.getData(i, 0)), 2));
	}

	public void learningrate() {
		lamda = 0.01 * Math.sqrt(t);
	}

	public void info() {
		System.out.println("\nError of Propagation " + t + " is:\n" + C.get(t - 1) + "\n");
		System.out.println("The resulting Matrix is: ");
		Matrix.printM(Layers[No_l - 1]);
		System.out.println("The result should be: ");
		Matrix.printM(y);
	}

	public void printAll() {
		for (int i = 0; i < No_l - 1; i++) {
			System.out.println("Layer " + i);
			System.out.println("weights: ");
			Matrix.printM(weights[i]);
			System.out.println("Bias: ");
			Matrix.printM(Bias[i]);

		}

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
