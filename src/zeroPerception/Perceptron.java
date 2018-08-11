package zeroPerception;

import java.util.ArrayList;
import java.util.List;

import zeroMath.Matrix;

/**
 * @author z3r0r4
 * @version 1.0
 */

public class Perceptron {
	private Matrix[] a, z, b, w;
	private Matrix[] Gradient_a, Gradient_w, Gradient_b;
	private int No_l;// number of Layers

	private Matrix y;
	private List<Double> C = new ArrayList<Double>();
	private List<Double> Cmean = new ArrayList<Double>();
	public int t = 0;
	private double lamda = 0.1;

	public Perceptron(int... NumberOfNodes) { //NumberOfNodes[Layer] Anzahl der nodes im layer 'Layer' 
		No_l = NumberOfNodes.length; //NumberOfNodes.length Anzahl der Layerrs
		System.out.println("+++++++++++initialising Variables for Forwardpropagation+++++++++++");
		System.out.println("generating Nodes foreach Layer");
		a = new Matrix[No_l];
		for (int i = 0; i < No_l; i++) {
			a[i] = new Matrix(NumberOfNodes[i], 1); //filled with 0

		}

		System.out.println("generating Inbetweenthingys, Biases and Weights foreach Layer except the last");
		z = new Matrix[No_l - 1];
		b = new Matrix[No_l - 1];
		w = new Matrix[No_l - 1];
		for (int i = 0; i < No_l - 1; i++) {
			System.out.println("intialising Layer " + i);
			z[i] = new Matrix(a[i + 1].getRows(), 1); // filled with 0
			b[i] = new Matrix(a[i + 1].getRows(), 1, 1, 10); // filled with random data
			w[i] = new Matrix(a[i + 1].getRows(), a[i].getRows(), 1, 10); // filled with random data
		}

		System.out.println("+++++++++++initialising Variables for Backpropagation+++++++++++");
		System.out.println("generating Node, Bias and Weight Gradients foreach Layer except the last");
		Gradient_a = new Matrix[No_l];
		Gradient_b = new Matrix[No_l - 1];
		Gradient_w = new Matrix[No_l - 1];
		for (int i = 0; i < No_l - 1; i++) {
			System.out.println("intialising Layer " + i);
			Gradient_a[i] = new Matrix(a[i].getRows(), 1); // filled with 0
			Gradient_b[i] = new Matrix(b[i].getRows(), 1); // filled with 0
			Gradient_w[i] = new Matrix(a[i + 1].getRows(), a[i].getRows()); // filled with 0
		}
		Gradient_a[No_l - 1] = new Matrix(a[No_l - 1].getRows(), 1); // filled with 0 // Node Gradient for the last layer

		System.out.println("-------------------------------------------------");
		System.out.println("generating correct Answers");
		y = new Matrix(a[a.length - 1].getRows(), 1);
		for (int i = 0; i < y.getRows(); i++)
			y.setData(10, i, 0); // filled with 0 for now atleast
		//y.info(null);

		System.out.println("++++++++++Finished Initialisation of variables+++++++++++++ \n\n");
	}

	public void forwardProp(double[][] input) {//takes the input for the network as input
		System.out.println("\n++++++++FORWARD-PROPAGATION++++++++++" + t);
		a[0].setData(input);
		for (int i = 1; i < No_l; i++) {
			//System.out.printf("calculating Layer %d \n", i);

			z[i - 1].setData(
					Matrix.add(
							Matrix.prod(w[i - 1], a[i - 1]),
							b[i - 1])
							.getData());

			a[i].setData(sigmoid(z[i - 1]).getData());
			//						a[i].setData(  		//cant be used since backprop needs z[n] defined
			//								sigmoid(
			//								Matrix.add(
			//										Matrix.prod(w[i - 1], a[i - 1]),
			//										b[i - 1])
			//										).getData());
		}
	}

	public void backProp(double[][] correctAnswer) {
		System.out.println("\n++++++++BACKWARD-PROPAGATION+++++++++" + t);
		//System.out.println("---CALCULATION OF a_GRADIENT FOR EVERY LAYER---");
		y.setData(correctAnswer);
		//Gradient computation for the last Layer
		for (int i = 0; i < Gradient_a[No_l - 1].getRows(); i++)

			Gradient_a[No_l - 1].setData(
					2 * (a[No_l - 1].getData(i, 0) - y.getData(i, 0)), //first derivative of the cost function
					i, 0);

		//Gradient computation for all layers except the last
		double var = 0;
		for (int n = No_l - 2; n > 0; n--) {
			for (int i = 0; i < Gradient_a[n].getRows(); i++) {
				var = 0;

				for (int l = 0; l < z[n].getRows(); l++) {
					var += w[n].getData(l, i)
							* sigmoid_1(z[n].getData(l, 0))
							* Gradient_a[n + 1].getData(l, 0);
				}
				Gradient_a[n].setData(var, i, 0);
			}
		}

		//System.out.println("---CALCULATION OF b_GRADIENT FOR EVERY LAYER---");

		for (int n = No_l - 2; n > 0; n--) { // -2 Because -1 => upper bound -2 => one lower than upper bound|| bound of b[n]
			for (int i = 0; i < Gradient_b[n].getRows(); i++) {
				Gradient_b[n].setData(
						sigmoid_1(z[n].getData(i, 0))
								* Gradient_a[n + 1].getData(i, 0),
						i, 0);
			}
		}
		//System.out.println("---CALCULATION OF w_GRADIENT FOR EVERY LAYER---");
		for (int n = No_l - 2; n > 0; n--) {
			Gradient_w[n].setData(
					Matrix.transpose(
							Matrix.prod(
									Matrix.mult(a[n], Gradient_a[n]),
									sigmoid_1(Matrix.transpose(z[n]))))
							.getData());
		}
	}

	public void forwardInfo() {
		for (int i = 0; i < No_l - 1; i++)
			System.out.printf("\n%d. Layer's | Nodes: %d & Z's: %d & biases: %d & weights %dx%d  ", i, a[i].getRows(),
					z[i].getRows(), b[i].getRows(), w[i].getRows(), w[i].getColumns());
		// info about last layer
		System.out.printf("%d. Layer's | Nodes: %d & y's: %d  \n", No_l - 1, a[No_l - 1].getRows(), y.getRows());
	}

	public void backwardInfo() {
		for (int i = 0; i < No_l - 1; i++)
			System.out.printf("\n%d. Layer's | Node Gradients: %d & Bias Gradients: %d & Weight Gradients %dx%d  \n", i,
					Gradient_a[i].getRows(), Gradient_b[i].getRows(), Gradient_w[i].getRows(),
					Gradient_w[i].getColumns());
	}

	public void cost() { //total error of one propagation
		double var = 0;
		for (int i = 0; i < a[No_l - 1].getRows(); i++)
			var += Math.pow((a[No_l - 1].getData(i, 0) - y.getData(i, 0)), 2);
		C.add(var);
	}

	public void meanCost(int l) { // total error over l different propagations (everytime different input/answers)
		double var = 0;
		for (int i = 0; i < l; i++)
			var += C.get(C.size() - i);
		Cmean.add(var / l);
	}

	public void learningrate() {
		lamda = 0.01 * Math.sqrt(t);
	}

	public void APPLY() {
		for (int n = No_l - 2; n > 0; n--) {
			b[n].add(Matrix.scalarmult(lamda, Matrix.negate(Gradient_b[n])));
			w[n].add(Matrix.scalarmult(lamda, Matrix.negate(Gradient_w[n])));
		}
	}

	public void info() {
		System.out.println("\nError of Propagation " + t + " is:\n" + C.get(t - 1));
		System.out.println("The resulting Matrix is: ");
		Matrix.printM(a[No_l - 1]);
	}

	public void printAll() {
		for (int i = 0; i < No_l - 1; i++) {
			System.out.println("Layer " + i);
			System.out.println("w: ");
			Matrix.printM(w[i]);
			System.out.println("b: ");
			Matrix.printM(b[i]);
			System.out.println("z: ");
			Matrix.printM(z[i]);
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

	private Matrix sigmoid_1(Matrix A) {
		Matrix B = new Matrix(A.getRows(), A.getColumns());
		for (int i = 0; i < A.getRows(); i++)
			for (int j = 0; j < A.getColumns(); j++)
				B.setData(sigmoid_1(A.getData()[i][j]), i, j);
		return B;
	}

	private double sigmoid_1(double a) {
		return sigmoid(a) * (1 - sigmoid(a));
	}
}
