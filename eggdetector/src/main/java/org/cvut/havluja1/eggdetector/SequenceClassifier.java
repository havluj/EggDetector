package org.cvut.havluja1.eggdetector;

import javax.imageio.ImageIO;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class SequenceClassifier {

    private Map<String, Integer> imageScores;
    private final TensorFlowObjectDetectionModel tensorFlowObjectDetectionModel = new TensorFlowObjectDetectionModel();

    SequenceClassifier(List<File> images, float minimalConfidence, boolean debugMode) {
        this.imageScores = new HashMap<>();
        for (File f : images) {
            imageScores.put(f.getName(), getImgScore(f, minimalConfidence, debugMode));
        }
    }

    public Integer getFinalCount() {
        TreeMap<Integer, Integer> scores = new TreeMap<>();

        for (Integer val : imageScores.values()) {
            if (scores.containsKey(val)) {
                scores.replace(val, scores.get(val) + 1); // increment
            } else {
                scores.put(val, 1);
            }
        }

        int bestGuess = 0;
        while (!scores.isEmpty()) {
            Map.Entry<Integer, Integer> e = scores.pollLastEntry();
            if (e.getValue() > 0) { // threshold (how many times do we need the value)
                return e.getKey();
            } else if (e.getValue() == 1) {
                bestGuess = e.getValue();
            }

        }

        return bestGuess;
    }

    public Map<String, Integer> getIndividualCounts() {
        return new HashMap<>(imageScores);
    }

    private int getImgScore(File imageFile, float minimalConfidence, boolean debugMode) {
        try {
            List<TensorFlowObjectDetectionModel.Recognition> recognitions =
                    tensorFlowObjectDetectionModel.recognizeImage(ImageIO.read(imageFile), minimalConfidence, debugMode);
            return recognitions.size();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0;
    }

}
