/**
 * 
 */
package zeroPerception;

/**
 * @author z3r0r4
 * 
 */
public class Main {

	static Perceptron A = new Perceptron(10,2, 1);
	static double[][] input = new double[10][1];
	static double[][] y = { { 0.5 } };

	public static void main(String[] args) {

		long lStartTime = System.nanoTime();

		for (int k = 0; k < 1; k++) {
			for (int i = 0; i < input.length; i++)
				input[i][0] = Math.random();
			forward();
			backward();
			A.cost();
		}

		long lEndTime = System.nanoTime();
		long output = lEndTime - lStartTime;
		double outputsec = output * Math.pow(10, -9);
		double outputmillisec = output * Math.pow(10, -6);
		System.out.println("Elapsed time in milliseconds: " + outputmillisec);
		System.out.println("Elapsed time in seconds: " + outputsec);
		System.out.println("Seconds needed for one Layer: " + outputsec / A.t);
		A.info();
		//A.printALL();

	}

	public static void forward() {
		A.forwardProp(input);
		//A.forwardInfo();
		A.t++;
	}

	public static void backward() {
		A.backProp(y);
		//A.backwardInfo();
		A.APPLY();
	}
}
