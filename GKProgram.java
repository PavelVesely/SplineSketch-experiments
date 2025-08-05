import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.*;

public class GKProgram {

    static double[] arr;
    static List<Double> data;

    public static void main(String[] args) {
        // NOTE: mergeability test not implemented here --- no merge op. in our implementation of GK
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java GKProgram <dataset_file> <query_file> <k> <output_file>");
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
            //arr = new double[data.size()];
            //for (int i = 0; i < data.size(); i++) {
            //    arr[i] = data.get(i);
            //}
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

            // Create GKSketch with the given compression parameter
            GK gk = new GK(3.0/(2*k)); // FIXME: which epsilon to choose?
            for (int i = 0; i < data.size(); i++) {
                gk.feed(data.get(i));
            }
            gk.finalize();
            long afterUpdatesTime = System.nanoTime();

            List<Long> results = new ArrayList<>();
            for (int i = 0; i < queries.size(); i++) {
                results.add((gk.query_for_rank(queries.get(i))));
            }
            long afterQueriesTime = System.nanoTime();
            int tuples = gk.num_entries_stored();

            // Print the size of the serialized sketch in bytes
            System.out.printf("%d%n", tuples * 24); // three 64-bit numbers per tuple in GK
            System.out.printf("%d%n", afterUpdatesTime - startTime);
            System.out.printf("%d%n", afterQueriesTime - afterUpdatesTime);
            
            try (PrintWriter outputWriter = new PrintWriter(new FileWriter(outputFile))) {
                if (results != null) {
                    for (int i = 0; i < results.size(); i++) {
                        outputWriter.printf("%d%n", results.get(i));
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (Exception e) {
            System.err.printf("GKSketchProgram exception %n" + e.getMessage());
            e.printStackTrace();
        }
    }

    

}
