package zeroPerception;

/**
 * @author z3r0r4
 * 
 */
public class Main {

	static Perceptron A = new Perceptron(2, 2, 3, 1);
	static double[][] input, correctAnswer;
	static long lStartTime, lEndTime;

	public static void main(String[] args) {
		lStartTime = System.nanoTime();

		System.out.println("Training");

		for (int i = 0; i < 100000; i++) {
			XORinit();
			A.Train(1, input, correctAnswer);
		}

		lEndTime = System.nanoTime();
		timer();

		input = new double[][] { { 1 }, { 1 } };
		correctAnswer = new double[][] { { 0 } };
		A.Train(1, input, correctAnswer);
		A.infoT();
	}

	//	public static void init() { //redo
	//		input = new double[2][1];
	//		correctAnswer = new double[1][1];
	//		input = new double[][] { { Math.random() }, { Math.random() } };
	//		if (input[1][0] < h(input[0][0]) && input[1][0] > l(input[0][0])) {
	//			correctAnswer = new double[][] { { 0.99 } };
	//		} else {
	//			correctAnswer = new double[][] { { 0.01 } };
	//		}
	//	}
	public static void XORinit() {
		input = new double[2][1];
		correctAnswer = new double[1][1];
		double a = Math.random() * 10;
		if (a < 2.5) {
			input = new double[][] { { 0 }, { 0 } };
			correctAnswer = new double[][] { { 0 } };
		} else if (a >= 2.5 && a < 5) {
			input = new double[][] { { 1 }, { 1 } };
			correctAnswer = new double[][] { { 0 } };
		} else if (a >= 5 && a < 7.5) {
			input = new double[][] { { 1 }, { 0 } };
			correctAnswer = new double[][] { { 1 } };
		} else if (a > 7.5) {
			input = new double[][] { { 0 }, { 1 } };
			correctAnswer = new double[][] { { 1 } };
		}
	}

	public static void timer() {
		double output, outputsec, outputmillisec;

		output = lEndTime - lStartTime;
		outputsec = output * Math.pow(10, -9);
		outputmillisec = output * Math.pow(10, -6);

		System.out.println("\nElapsed time in milliseconds: " + outputmillisec);
		System.out.println("Elapsed time in seconds: " + outputsec);
		System.out.println("Seconds needed for one Iteration: " + outputsec / A.getNumber_of_iterations());
	}
}
