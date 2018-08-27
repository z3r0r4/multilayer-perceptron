package zeroPerception;

import java.util.ArrayList;

public class Data {//do this with an generic pair

	public ArrayList<double[]> inputData;
	public ArrayList<double[]> targetData;

	public Data(ArrayList<double[]> x, ArrayList<double[]> y) {
		this.inputData = x;
		this.targetData = y;
	}

}
