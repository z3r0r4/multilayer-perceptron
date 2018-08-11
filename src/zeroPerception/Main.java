/**
 * 
 */
package zeroPerception;

/**
 * @author z3r0r4
 * 
 */
public class Main {

	static Perceptron A = new Perceptron(2, 3,2, 1);
	static double[][] input = new double[2][1];
	static double[][] y = new double[1][1];//{ { 1 }, { 0 }, { 0 }, { 1 }, { 0 }, { 0 }, { 1 }, { 0 }, { 0 } };
	static long lStartTime, lEndTime;

	public static void main(String[] args) {

		lStartTime = System.nanoTime();

		for (int k = 0; k < 50000; k++) {
			input = new double[][] { { Math.random() }, { Math.random() } };
			if (input[1][0] < ge(input[0][0]) && input[1][0] > ef(input[0][0])) {
				y = new double[][] { { 0.99 } };
			} else {
				y = new double[][] { { 0.01 } };
			}
			//	genTrain();
			forward();
			backward();
			A.cost();
		}

		lEndTime = System.nanoTime();

		time();
		A.info();
		System.out.println("tyyyyhhhhhhhhhhhhhh");
		input = new double[][] { { 0 }, { 0 } };
		y = new double[][] {{0.01} };
		forward();
		A.cost();
		A.info();
		input = new double[][] { { 0 }, { 0 } };
		y = new double[][] {{0.01} };
		forward();
		A.cost();
		A.info();

		//A.printAll();
	}

	public static void forward() {
		A.t++;
		A.forwardProp(input);
		System.out.println(y[0][0]);
		//A.forwardInfo();

	}

	public static void backward() {
		A.backProp(y);
		//A.backwardInfo();
		A.APPLY();
	}

	public static void time() {

		long output = lEndTime - lStartTime;
		double outputsec = output * Math.pow(10, -9);
		double outputmillisec = output * Math.pow(10, -6);
		System.out.println("Elapsed time in milliseconds: " + outputmillisec);
		System.out.println("Elapsed time in seconds: " + outputsec);
		System.out.println("Seconds needed for one Layer: " + outputsec / A.t);
	}

	
	
	

	public static double ge(double x) {
		return -(x - 1) + 0.5;
	}

	public static double ef(double x) {
		return -x + 0.5;
	}

}


