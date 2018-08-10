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
		Perceptron A = new Perceptron(5);
		A.forwardProp();
		//A.forwardInfo();
		A.backProp();
		//A.backwardInfo();
		//A.printALL();
		A.APPLY();
	}

}
