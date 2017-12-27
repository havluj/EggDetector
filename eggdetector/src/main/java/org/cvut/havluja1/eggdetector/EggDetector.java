package org.cvut.havluja1.eggdetector;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.apache.commons.io.FilenameUtils;

public class EggDetector {

    private final static Logger LOGGER = Logger.getLogger(EggDetector.class.getName());
    private float minimalConfidence = 0.2f;
    private boolean debugMode = false;

    public EggDetector() {
        LOGGER.info("loading TensorFlow graph into memory...");
        TensorFlowInferenceInterface.getInstance();
        LOGGER.info("EggDetector ready: " + toString());
    }

    public float getMinimalConfidence() {
        return minimalConfidence;
    }

    public void setMinimalConfidence(float minimalConfidence) {
        LOGGER.info("setting minimum confidence to: " + minimalConfidence);
        this.minimalConfidence = minimalConfidence;
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public void setDebugMode(boolean debugMode) {
        LOGGER.info("setting debug mode to " + debugMode);
        this.debugMode = debugMode;
    }

    public void closeSession() {
        LOGGER.info("closing session");
        TensorFlowInferenceInterface.getInstance().close();
    }

    public SequenceClassifier evaluate(File dir) throws IllegalArgumentException {
        LOGGER.info("evaluating dir: " + dir.getAbsolutePath());
        if (!dir.isDirectory()) {
            throw new IllegalArgumentException(dir.getAbsolutePath() + " is not a directory.");
        }

        List<File> imageList = Arrays.asList(dir.listFiles((File file, String name) -> {
            File workingFile = new File(file.getAbsolutePath() + File.separator + name);
            return workingFile.isFile()
                    && (FilenameUtils.getExtension(name).equals("png")
                    || FilenameUtils.getExtension(name).equals("jpg"));
        }));

        if (imageList.isEmpty()) {
            throw new IllegalArgumentException(dir.getAbsolutePath() + " does not contain any pictures.");
        }
        LOGGER.info(imageList.size() + " pictures found");

        return new SequenceClassifier(imageList, minimalConfidence, debugMode);
    }

    @Override
    public String toString() {
        return "EggDetector{" +
                "minimalConfidence=" + minimalConfidence +
                ", debugMode=" + debugMode +
                '}';
    }
}
