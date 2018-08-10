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
		A.ForwardInfo();
		A.ForwardProp();
		A.BackwardInfo();
		A.BackProp();
		//A.printALL();
		A.APPLY();
	}

}
