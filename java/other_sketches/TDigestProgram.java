import com.tdunning.math.stats.TDigest;
import com.tdunning.math.stats.ScaleFunction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Arrays;

public class TDigestProgram {

    public static void main(String[] args) {
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java TDigestProgram <dataset_file> <query_file> <compression> <output_file> <num_parts (optional)>");
            return;
        }

        String datasetFile = args[0];
        String queryFile = args[1];
        double compression = Double.parseDouble(args[2]);
        String outputFile = args[3];
        int num_parts = (args.length == 5) ? Integer.parseInt(args[4]) : 1; // streaming if it's 1, mergeability otherwise or merge

        try {
            ////////////// load data and queries /////////////////
            List<Double> data = new ArrayList<>();
            // Read the dataset file and add numbers to the SplineSketch
            try (BufferedReader datasetReader = new BufferedReader(new FileReader(datasetFile))) {
                String line;
                while ((line = datasetReader.readLine()) != null) {
                    double value = Double.parseDouble(line);
                    data.add(value);
                }
            }
            int n = data.size();
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
            TDigest tDigest;
            if (num_parts == 1) { // streaming
                startTime = System.nanoTime();
                // Create a TDigest with the given compression parameter
                tDigest = TDigest.createDigest(compression); // we use the default MergingDigest
                tDigest.setScaleFunction(ScaleFunction.K_0);

                for (int i = 0; i < data.size(); i++) {
                    tDigest.add(data.get(i));
                }
            }
            else {
                int partSize = n / num_parts; // somewhat assuming this will be integer (no remainder)
                TDigest[] sketches = new TDigest[num_parts];
                for (int j = 0; j < num_parts; j++) {
                    sketches[j] = TDigest.createDigest(compression); // we use the default MergingDigest
                    sketches[j].setScaleFunction(ScaleFunction.K_0);
                }
                // create individual sketches
                int j = 0;
                for (int i = 0; i < n; i++) {
                    sketches[j].add(data.get(i));
                    if (i % partSize == partSize - 1 && j < num_parts - 1) j++;
                }
                ////////////// measure time from here /////////////////
                startTime = System.nanoTime();
                // merging
                for (int step = 1; step < num_parts; step *= 2) {
                    for (j = 0; j < num_parts - step; j += 2*step) {
                        sketches[j].add(sketches[j+step]); // merge operation
                        sketches[j+step] = null;
                    }
                }
                tDigest = sketches[0];

                //assert .getN() == n;
            }
            tDigest.compress();
            afterUpdatesTime = System.nanoTime();

            List<Integer> results = new ArrayList<>();
            for (int i = 0; i < queries.size(); i++) {
                results.add((int)(tDigest.cdf(queries.get(i)) * n));
            }
            afterQueriesTime = System.nanoTime();

            try (PrintWriter outputWriter = new PrintWriter(new FileWriter(outputFile))) {
                for (int i = 0; i < results.size(); i++) {
                    outputWriter.printf("%d%n", results.get(i));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            // Print the size of the sketch
            System.out.printf("%d%n", tDigest.byteSize());
            System.out.printf("%d%n", afterUpdatesTime - startTime);
            System.out.printf("%d%n", afterQueriesTime - afterUpdatesTime);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
