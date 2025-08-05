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
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java MomentSketchProgram <dataset_file> <query_file> <k> <output_file> <num_parts (optional)>");
            return;
        }

        String datasetFile = args[0];
        String queryFile = args[1];
        int k = Integer.parseInt(args[2]);
        String outputFile = args[3];
        int num_parts = (args.length == 5) ? Integer.parseInt(args[4]) : 1; // streaming if it's 1, mergeability otherwise or merge

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
            long startTime, afterUpdatesTime, afterQueriesTime;
            CMomentSketch  ms;
            ////////////// measure time from here /////////////////
            startTime = System.nanoTime();

            if (num_parts == 1) { // streaming
                // Create MomentSketch with the given compression parameter
                ms = new CMomentSketch(1e-9); // use default tolerance
                ms.setSizeParam(k);
                ms.initialize();
                ms.add(arr);
            } else {
                int partSize = n / num_parts; // somewhat assuming this will be integer (no remainder)
                CMomentSketch[] sketches = new CMomentSketch[num_parts];
                int i = 0;
                // create individual sketches
                for (int j = 0; j < num_parts; j++) {
                    sketches[j] = new CMomentSketch(1e-9); // use default tolerance
                    sketches[j].setSizeParam(k);
                    sketches[j].initialize();
                    double[] sub = Arrays.copyOfRange(arr, i, Math.min(i+partSize, n));
                    i += partSize;
                    sketches[j].add(sub);
                }
                ////////////// measure time from here /////////////////
                startTime = System.nanoTime();
                // merging
                for (int step = 1; step < num_parts; step *= 2) {
                    for (int j = 0; j < num_parts - step; j += 2*step) {
                        sketches[j].merge(new ArrayList<>(Collections.singletonList(sketches[j+step])), 0, 1);
                        sketches[j+step] = null;
                    }
                }
                ms = sketches[0];

            }


            afterUpdatesTime = System.nanoTime();

            Double[] results = getRanks(ms, n, queries); 
            afterQueriesTime = System.nanoTime();

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