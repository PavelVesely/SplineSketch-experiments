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
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java GKProgram <dataset_file> <query_file> <k> <output_file> <num_parts (optional)>");
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

            GK gk;
            if (num_parts == 1) { // streaming
                // Create GKSketch with the given compression parameter
                gk = new GK(3.0/(2*k)); // empirically a good choice of epsilon
                for (int i = 0; i < data.size(); i++) {
                    gk.feed(data.get(i));
                }
                gk.finalize();
            }
            else {
                int partSize = n / num_parts; // somewhat assuming this will be integer (no remainder)
                GK[] sketches = new GK[num_parts];
                for (int j = 0; j < num_parts; j++) {
                    sketches[j] = new GK(3.0/(2*k));
                }
                // create individual sketches
                int j = 0;
                for (int i = 0; i < n; i++) {
                    sketches[j].feed(data.get(i));
                    if (i % partSize == partSize - 1 && j < num_parts - 1) j++;
                }
                ////////////// measure time from here /////////////////
                startTime = System.nanoTime();
                // merging
                for (int step = 1; step < num_parts; step *= 2) {
                    for (j = 0; j < num_parts - step; j += 2*step) {
                        sketches[j].merge(sketches[j+step]);
                        sketches[j+step] = null;
                    }
                }
                gk = sketches[0];
            }
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
