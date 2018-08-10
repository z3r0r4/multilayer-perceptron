/**
 * 
 */
package zeroPerception;

/**
 * @author z3r0r4
 * 
 */
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		double[][] B = { { 1 } };
		double[][] C = { { 1 } };
		Perceptron A = new Perceptron(1);
		A.forwardProp(B);
		//A.forwardInfo();
		A.backProp(C);
		//A.backwardInfo();
		A.APPLY();
		//A.cost();
		//A.info();
		//A.printALL();
	}

}
