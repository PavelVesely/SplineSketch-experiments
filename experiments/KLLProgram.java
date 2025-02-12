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
        if (args.length != 4) {
            System.out.println("Usage: java KLLProgram <dataset_file> <query_file> <k> <output_file>");
            return;
        }

        String datasetFile = args[0];
        String queryFile = args[1];
        Integer k = Integer.parseInt(args[2]);
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
            
            long startTime = System.nanoTime();

            // Create a KLL with given 
            org.apache.datasketches.kll.KllDoublesSketch kll = org.apache.datasketches.kll.KllDoublesSketch.newHeapInstance(k);
            for (int i = 0; i < data.size(); i++) {
                kll.update(data.get(i), 1);
            }
            long afterUpdatesTime = System.nanoTime();

            double[] results = kll.getRanks(queries, org.apache.datasketches.quantilescommon.QuantileSearchCriteria.INCLUSIVE);
            // for (int i = 0; i < queries.length; i++) {
            //     results.add(kll.getRank(queries.get(i), org.apache.datasketches.quantilescommon.QuantileSearchCriteria.INCLUSIVE));
            // }
            long afterQueriesTime = System.nanoTime();
            
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
