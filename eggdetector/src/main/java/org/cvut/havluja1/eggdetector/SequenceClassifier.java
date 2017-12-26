package org.cvut.havluja1.eggdetector;

import javax.imageio.ImageIO;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SequenceClassifier {

    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.4f;

    private Map<String, Integer> imageScores;
    private final TensorFlowObjectDetectionModel tensorFlowObjectDetectionModel = new TensorFlowObjectDetectionModel();

    SequenceClassifier(List<File> images) {
        this.imageScores = new HashMap<>();

        String imageLoc = "/home/jan/bachelor_thesis/object-detection-training/test_images/image4.png"; // 1280x720 test image
//        floatValues = new float[1280 * 720 * 3];
        System.out.println(getImgScore(new File(imageLoc)));
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

    private int getImgScore(File imageFile) {
        try {
            List<Classifier.Recognition> recognitions = tensorFlowObjectDetectionModel.recognizeImage(ImageIO.read(imageFile));
            System.out.println(recognitions);
            return recognitions.size();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0;
    }

}
