import com.datadoghq.sketch.ddsketch.DDSketch;
import com.datadoghq.sketch.ddsketch.DDSketches;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.*;

public class DDSketchProgram {

    static double[] arr;
    static List<Double> data;

    public static void main(String[] args) {
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java DDSketchProgram <dataset_file> <query_file> <k> <output_file> <num_parts (optional)>");
            return;
        }

        // FIXME: DDSketch size apparently depends on whether the input contains negative number or not

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

            double a = 0.5 / k; // TODO how to set alpha???
            long startTime, afterUpdatesTime, afterQueriesTime;
            DDSketch sketch;
            if (num_parts == 1) { // streaming
                ////////////// measure time from here /////////////////
                startTime = System.nanoTime();

                sketch = DDSketches.collapsingLowestDense(a, k);
                for (int i = 0; i < data.size(); i++) {
                    sketch.accept(data.get(i));
                }
            } else {
                int partSize = n / num_parts; // somewhat assuming this will be integer (no remainder)
                DDSketch[] sketches = new DDSketch[num_parts];
                for (int j = 0; j < num_parts; j++) {
                    sketches[j] = DDSketches.collapsingLowestDense(a, k);;
                }
                // create individual sketches
                int j = 0;
                for (int i = 0; i < n; i++) {
                    sketches[j].accept(data.get(i));
                    if (i % partSize == partSize - 1 && j < num_parts - 1) j++;
                }
                ////////////// measure time from here /////////////////
                startTime = System.nanoTime();
                // merging
                for (int step = 1; step < num_parts; step *= 2) {
                    for (j = 0; j < num_parts - step; j += 2*step) {
                        sketches[j].mergeWith(sketches[j+step]);
                        sketches[j+step] = null;
                    }
                }
                sketch = sketches[0];

            }

            assert sketch.getCount() == n;


            afterUpdatesTime = System.nanoTime();

            double[] results = getRanks(sketch, n, queries); 
            afterQueriesTime = System.nanoTime();

            // Print the size of the serialized sketch in bytes
            System.out.printf("%d%n", sketch.serializedSize());
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

    private static double[] getRanks(DDSketch sketch, int n, List<Double> queries) {
        int m = queries.size();
        double[] qL = new double[m];
        double[] qR = new double[m];
        double[] q = new double[m];
        for (int i = 0; i < m; i++) {
            qL[i] = 0.0;
            qR[i] = 1.0;
            q[i] = 0.5;
        }
        try {
            while ((qR[0] - qL[0]) * n > 1) {
                    double[] res = sketch.getValuesAtQuantiles(q); // a lot of repeated queries... but not clear what to do with it
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