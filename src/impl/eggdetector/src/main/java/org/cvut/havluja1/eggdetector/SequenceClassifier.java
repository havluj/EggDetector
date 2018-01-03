package org.cvut.havluja1.eggdetector;

import javax.imageio.ImageIO;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * <h1>A class containing object detection results for a given directory</h1>
 * <p>SequenceClassifier is a data class containing the results of object
 * detection for a given directory. When constructed, object detection is performed
 * on all images and results are stored in memory.</p>
 * <p><b>Example usage:</b></p>
 * <pre>
 * {@code
 * EggDetector eggDetector = new EggDetector();
 * SequenceClassifier sequenceClassifier = eggDetector.evaluate(new File("image_dir"));
 * System.out.println("final count: " + sequenceClassifier.getFinalCount());
 * System.out.println("individual scores: " + sequenceClassifier.getIndividualCounts());
 * eggDetector.closeSession();
 * }
 * </pre>
 *
 * @author Jan Havluj {@literal <jan@havluj.eu>}
 * @version 1.0
 */
public class SequenceClassifier {

    private final TensorFlowObjectDetectionModel tensorFlowObjectDetectionModel = new TensorFlowObjectDetectionModel();
    private Map<String, Integer> imageScores;

    SequenceClassifier(List<File> images, float minimalConfidence, boolean debugMode) {
        this.imageScores = new HashMap<>();
        for (File f : images) {
            imageScores.put(f.getName(), getImgScore(f, minimalConfidence, debugMode));
        }
    }

    /**
     * <p>Get the final score for the entire directory.</p>
     * <p>The final score is calculated as follows:</p>
     * <ul>
     * <li>individual scores of images are sorted and counted</li>
     * <li>the highest egg count is returned as a result if we detected this
     * egg count in at least two different images</li>
     * <li>if no two images contain the same egg count, the highest detected
     * egg count is returned</li>
     * <li>if no eggs are detected in any of the images, 0 is returned</li>
     * </ul>
     *
     * @return final egg count for this instance
     */
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
            if (e.getValue() > 1) { // threshold (how many times do we need the value)
                return e.getKey();
            } else if (e.getValue() == 1) {
                bestGuess = e.getValue();
            }

        }

        return bestGuess;
    }

    /**
     * Gets the individual egg count for every image provided.
     *
     * @return A map of individual scores. The key is the filename. The value is the egg count.
     */
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
