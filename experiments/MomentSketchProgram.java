import sketches.CMomentSketch;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.*;

public class MomentSketchProgram {

    static double[] arr;
    static List<Double> data;

    public static void main(String[] args) {
        if (args.length != 4) {
            System.out.println("Usage: java MomentSketchProgram <dataset_file> <query_file> <k> <output_file>");
            return;
        }

        String datasetFile = args[0];
        String queryFile = args[1];
        int k = Integer.parseInt(args[2]);
        String outputFile = args[3];

        try {
            ////////////// load data and queries /////////////////
            data = new ArrayList<>();
            // Read the dataset file and add numbers to the SplineSketch
            try (BufferedReader datasetReader = new BufferedReader(new FileReader(datasetFile))) {
                String line;
                while ((line = datasetReader.readLine()) != null) {
                    double value = Double.parseDouble(line);
                    data.add(value);
                }
            }
            int n = data.size();
            arr = new double[data.size()];
            for (int i = 0; i < data.size(); i++) {
                arr[i] = data.get(i);
            }
            // Read the query file and query the SplineSketch using the cdf method
            List<Double> queries = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(queryFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        double value = Double.parseDouble(line);
                        queries.add(value);
                    } catch (NumberFormatException e) {
                        System.err.println("Skipping invalid float value: " + line);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            
            ////////////// measure time from here /////////////////
            long startTime = System.nanoTime();

            // Create MomentSketch with the given compression parameter
            CMomentSketch  ms = new CMomentSketch(1e-9); // use default tolerance
            ms.setSizeParam(k);
            ms.initialize();
            ms.add(arr);
            long afterUpdatesTime = System.nanoTime();

            Double[] results = getRanks(ms, n, queries); 
            long afterQueriesTime = System.nanoTime();

            // Print the size of the serialized sketch in bytes
            System.out.printf("%d%n", (2*k + 2)*8);
            System.out.printf("%d%n", afterUpdatesTime - startTime);
            System.out.printf("%d%n", afterQueriesTime - afterUpdatesTime);
            
            try (PrintWriter outputWriter = new PrintWriter(new FileWriter(outputFile))) {
                if (results != null) {
                    for (int i = 0; i < results.length; i++) {
                        outputWriter.printf("%d%n", (int)(results[i] * n));
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (Exception e) {
            System.err.printf("MomentSketchProgram exception %n" + e.getMessage());
            e.printStackTrace();
        }
    }

    private static Double[] getRanks(CMomentSketch ms, int n, List<Double> queries) {
        int m = queries.size();
        Double[] qL = new Double[m];
        Double[] qR = new Double[m];
        Double[] q = new Double[m];
        for (int i = 0; i < m; i++) {
            qL[i] = 0.0;
            qR[i] = 1.0;
            q[i] = 0.5;
        }
        try {
            while ((qR[0] - qL[0]) * n > 1) {
                    double[] res = ms.getQuantiles(Arrays.asList(q)); // a lot of repeated queries... but not clear what to do with it
                    for (int i = 0; i < m; i++) {
                        if (res[i] > queries.get(i)) {
                            qR[i] = q[i];
                        }
                        else {
                            qL[i] = q[i];
                        }
                        q[i] = (qL[i] + qR[i]) / 2;
                    }
            }
            return q;
        } catch (Exception e) {
            e.printStackTrace(System.err);
            return null; // will cause high error but at least we get something
        }
    }
}