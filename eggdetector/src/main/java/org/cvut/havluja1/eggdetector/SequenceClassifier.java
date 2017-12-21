package org.cvut.havluja1.eggdetector;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Operation;

public class SequenceClassifier {

    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

    private Map<String, Integer> imageScores;
    private final Graph graph;

    // Only return this many results.
    private static final int MAX_RESULTS = 100;

    // Config values.
    private String inputName;
    private int inputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private byte[] byteValues;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    SequenceClassifier(List<String> images, Graph graph) {
        this.graph = graph;

        inferenceInterface = new TensorFlowInferenceInterface(modelFilename);

        final Graph g = d.inferenceInterface.graph();

        inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + inputName + "'");
        }
        inputSize = inputSize;
        // The outputScoresName node has a shape of [N, NumLocations], where N
        // is the batch size.
        final Operation outputOp1 = g.operation("detection_scores");
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_scores'");
        }
        final Operation outputOp2 = g.operation("detection_boxes");
        if (outputOp2 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_boxes'");
        }
        final Operation outputOp3 = g.operation("detection_classes");
        if (outputOp3 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_classes'");
        }

        // Pre-allocate buffers.
        outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        intValues = new int[inputSize * inputSize];
        byteValues = new byte[inputSize * inputSize * 3];
        outputScores = new float[MAX_RESULTS];
        outputLocations = new float[MAX_RESULTS * 4];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];


        this.imageScores = new HashMap<>();
    }

    public Integer getFinalCount() {
        int maxEggCount = 0;

        for (Integer val : imageScores.values()) {
            if (val > maxEggCount) {
                maxEggCount = val;
            }
        }

        return maxEggCount;
    }

    public Map<String, Integer> getIndividualCounts() {
        return new HashMap<>(imageScores);
    }

    private int getImgScore(byte[] image) {

    }

}
