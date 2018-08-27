package zeroPerception;

import java.util.ArrayList;

public class XorData extends Data {

	public XorData() {
		super(input(), target());
	}

	public static ArrayList<double[]> input() {
		ArrayList<double[]> input;
		input = new ArrayList<>();
		input.add(new double[] { 0.01, 0.01 });
		input.add(new double[] { 0.99, 0.99 });
		input.add(new double[] { 0.99, 0.01 });
		input.add(new double[] { 0.01, 0.99 });
		return input;
	}

	public static ArrayList<double[]> target() {
		ArrayList<double[]> target;
		target = new ArrayList<>();
		target.add(new double[] { 0.01 });
		target.add(new double[] { 0.01 });
		target.add(new double[] { 0.99 });
		target.add(new double[] { 0.99 });
		return target;
	}

}
