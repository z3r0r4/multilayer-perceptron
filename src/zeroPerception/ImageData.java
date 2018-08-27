package zeroPerception;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ImageData extends Data {

	static ArrayList<double[]> Input, Target;

	public ImageData(String Path) {
		super(input(Path), target());
	}

	public static ArrayList<double[]> input(String Path) {
		Input = processInputFile(Path);
		System.out.println("+STARTING formatting input");
		Target = (ArrayList<double[]>) Input.clone();
		Input.replaceAll(array -> Arrays.copyOfRange(array, 1, array.length));
		int n = 0;
		for (double[] in : Input) {
			Arrays.stream(in).map(i -> i / 255.0 * 0.99 + 0.01);
			Input.set(n, in);
			n++;
		}
		System.out.println("-FINISHED formatting input");

		return Input;
	}

	public static ArrayList<double[]> target() {
		System.out.println("+STARTING formatting Target");
		double number = 0;
		Target.replaceAll(array -> Arrays.copyOfRange(array, 0, 1));
		int n = 0;
		for (double[] Tar : Target) {

			number = Tar[0];
			Tar = new double[10];
			Arrays.fill(Tar, 0.01);
			Tar[(int) number]=0.99;
//			for (int i = 0; i < 10; i++) {
//				Tar[i] = (i == number) ? 0.99 : 0.0;
//			}
			Target.set(n, Tar);
			n++;

		}
		System.out.println("-FINISHED formatting target");

		return Target;
	}

	private static ArrayList<double[]> processInputFile(String inputFilePath) { //https://dzone.com/articles/how-to-read-a-big-csv-file-with-java-8-and-stream
		System.out.println("+STARTING reading File");
		List<double[]> inputList = new ArrayList<double[]>();

		try {

			File inputF = new File(inputFilePath);

			InputStream inputFS = new FileInputStream(inputF);

			BufferedReader br = new BufferedReader(new InputStreamReader(inputFS));

			inputList = br.lines().map(mapToItem).collect(Collectors.toList());

			br.close();

		} catch (IOException e) {

			System.out.println(e.getMessage());

		}

		ArrayList<double[]> input = new ArrayList<>(inputList);
		System.out.println("-FINISHED reading File");
		return input;

	}

	private static Function<String, double[]> mapToItem = line -> {

		String[] p = line.split(",");// a CSV has comma separated lines

		double[] item = new double[p.length];
		int i = 0;
		for (String value : p) {

			item[i] = Double.parseDouble(value);
			i++;
		}
		return item;

	};
}
