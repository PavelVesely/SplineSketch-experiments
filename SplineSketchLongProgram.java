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
import java.util.Random;

public class SplineSketchLongProgram {

    public static void main(String[] args) {
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java SplineSketchLongProgram <dataset_file> <query_file> <sketch_size> <output_file> <num_parts (optional)>");
            return;
        }

        String datasetFile = args[0];
        String queryFile = args[1];
        int sketch_size = Integer.parseInt(args[2]);
        String outputFile = args[3];
        int num_parts = (args.length == 5) ? Integer.parseInt(args[4]) : 1; // streaming if it's 1, mergeability otherwise or merge

        try {
            ////////////// load data and queries /////////////////
            List<Long> data = new ArrayList<>();
            // Read the dataset file and add numbers to the SplineSketch
            try (BufferedReader datasetReader = new BufferedReader(new FileReader(datasetFile))) {
                String line;
                while ((line = datasetReader.readLine()) != null) {
                    long value = (long)Double.parseDouble(line);
                    data.add(value);
                }
            }
            int n = data.size();
            // Read the query file and query the SplineSketch using the cdf method
            List<Long> queries = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(queryFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        long value = (long)Double.parseDouble(line);
                        queries.add(value);
                    } catch (NumberFormatException e) {
                        System.err.println("Skipping invalid float value: " + line);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            long startTime, afterUpdatesTime, afterQueriesTime;
            SplineSketchLong splineSketch;
            if (num_parts == 1) { // streaming
            
                ////////////// measure time from here /////////////////
                startTime = System.nanoTime();

                // Create a SplineSketchLong with the given size
                splineSketch = new SplineSketchLong(sketch_size, "");
                for (int i = 0; i < data.size(); i++) {
                    splineSketch.update(data.get(i));
                }
            } else {
                int partSize = n / num_parts; // somewhat assuming this will be integer (no remainder)
                SplineSketchLong[] splineSketches = new SplineSketchLong[num_parts];
                for (int j = 0; j < num_parts; j++) {
                    splineSketches[j] = new SplineSketchLong(sketch_size, "");
                }
                // create individual sketches
                int j = 0;
                for (int i = 0; i < n; i++) {
                    splineSketches[j].update(data.get(i));
                    if (i % partSize == partSize - 1 && j < num_parts - 1) j++;
                }
                ////////////// measure time from here /////////////////
                startTime = System.nanoTime();
                // merging
                for (int step = 1; step < num_parts; step *= 2) {
                    for (j = 0; j < num_parts - step; j += 2*step) {
                        splineSketches[j] = SplineSketchLong.merge(splineSketches[j], splineSketches[j+step]);
                        splineSketches[j+step] = null;
                    }
                }
                splineSketch = splineSketches[0];

            }
            splineSketch.consolidate();
            assert splineSketch.getN() == n;

            ///////////////// measure time up to here ////////////
            afterUpdatesTime = System.nanoTime();
            List<Integer> result = splineSketch.query(queries);

            afterQueriesTime = System.nanoTime();

            try (PrintWriter outputWriter = new PrintWriter(new FileWriter(outputFile))) {
                for (int i = 0; i < result.size(); i++) {
                    outputWriter.printf("%d%n", result.get(i));                    
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            // Print the size of the sketch
            System.out.printf("%d%n", splineSketch.serializedSketchBytesCompact());
            System.out.printf("%d%n", afterUpdatesTime - startTime);
            System.out.printf("%d%n", afterQueriesTime - afterUpdatesTime);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
