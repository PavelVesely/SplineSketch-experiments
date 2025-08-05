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


public class KLLProgram {

    public static void main(String[] args) {
        if (args.length < 4 || args.length > 5) {
            System.out.println("Usage: java KLLProgram <dataset_file> <query_file> <k> <output_file> <num_parts (optional)>");
            return;
        }

        String datasetFile = args[0];
        String queryFile = args[1];
        Integer k = Integer.parseInt(args[2]);
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
            List<Double> queriesLst = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(queryFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        double value = Double.parseDouble(line);
                        queriesLst.add(value);
                    } catch (NumberFormatException e) {
                        System.err.println("Skipping invalid float value: " + line);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            double[] queries = new double[queriesLst.size()];
            for (int i = 0; i < queries.length; i++) {
                queries[i] = queriesLst.get(i);
            }
            
            long startTime, afterUpdatesTime, afterQueriesTime;
            org.apache.datasketches.kll.KllDoublesSketch kll;

            startTime = System.nanoTime();

            if (num_parts == 1) { // streaming
                // Create a KLL with given 
                kll = org.apache.datasketches.kll.KllDoublesSketch.newHeapInstance(k);
                for (int i = 0; i < data.size(); i++) {
                    kll.update(data.get(i), 1);
                }
            } else {
                int partSize = n / num_parts; // somewhat assuming this will be integer (no remainder)
                org.apache.datasketches.kll.KllDoublesSketch[] sketches = new org.apache.datasketches.kll.KllDoublesSketch[num_parts];
                for (int j = 0; j < num_parts; j++) {
                    sketches[j] = org.apache.datasketches.kll.KllDoublesSketch.newHeapInstance(k);
                }
                // create individual sketches
                int j = 0;
                for (int i = 0; i < n; i++) {
                    sketches[j].update(data.get(i), 1);
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
                kll = sketches[0];

            }
            assert kll.getN() == n;


            afterUpdatesTime = System.nanoTime();
            double[] results = kll.getRanks(queries, org.apache.datasketches.quantilescommon.QuantileSearchCriteria.INCLUSIVE);
            // for (int i = 0; i < queries.length; i++) {
            //     results.add(kll.getRank(queries.get(i), org.apache.datasketches.quantilescommon.QuantileSearchCriteria.INCLUSIVE));
            // }
            afterQueriesTime = System.nanoTime();
            
            try (PrintWriter outputWriter = new PrintWriter(new FileWriter(outputFile))) {
                for (int i = 0; i < results.length; i++) {
                    outputWriter.printf("%d%n", (int)(results[i] * n));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            // Print the size of the serialized sketch in bytes
            System.out.printf("%d%n", kll.getSerializedSizeBytes());
            System.out.printf("%d%n", afterUpdatesTime - startTime);
            System.out.printf("%d%n", afterQueriesTime - afterUpdatesTime);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
