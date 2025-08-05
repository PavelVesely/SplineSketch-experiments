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

public class SplineSketchMGProgram {

    public static void main(String[] args) {
        // NOTE: mergeability test not implemented here
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java SplineSketchMGProgram <dataset_file> <query_file> <sketch_size> <output_file>");
            return;
        }

        String datasetFile = args[0];
        String queryFile = args[1];
        int sketch_size = Integer.parseInt(args[2]);
        String outputFile = args[3];

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

            // Create a SplineSketch with the given size
            SplineSketchMG splineSketch = new SplineSketchMG(sketch_size, "");
            for (int i = 0; i < data.size(); i++) {
                splineSketch.update(data.get(i));
            }

            splineSketch.compressNonFrequentToBucketsAndResize();
            long afterUpdatesTime = System.nanoTime();
            ///////////////// measure time up to here ////////////

            List<Integer> result = splineSketch.query(queries);

            long afterQueriesTime = System.nanoTime();

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
