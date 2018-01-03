package org.cvut.havluja1.eggdetector;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.apache.commons.io.FilenameUtils;

/**
 * <h1>Count the number of eggs in given images</h1>
 * <p>The egg detector is a library that helps you count the number
 * of eggs in a given folder.</p>
 * <p>Egg detector works by using TensorFlow Object Detection API
 * in the background. To learn more, see
 * <a href="https://www.tensorflow.org/">https://www.tensorflow.com</a>.</p>
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
public class EggDetector {

    private final static Logger LOGGER = Logger.getLogger(EggDetector.class.getName());
    private float minimalConfidence = 0.3f;
    private boolean debugMode = false;
    private boolean sessionClosed = true;

    /**
     * <p>Constructor loads the pre-trained frozen graph into memory.</p>
     * <p>It also checks whether TensorFlow is supported on your platform.</p>
     */
    public EggDetector() {
        LOGGER.info("loading TensorFlow graph into memory...");
        try {
            TensorFlowInferenceInterface.getInstance();
            sessionClosed = false;
        } catch (Exception e) {
            LOGGER.severe("Error initializing TensorFlow");
            throw e;
        }
        LOGGER.info("EggDetector ready: " + toString());
    }

    /**
     * <p>Get the <i>minimalConfidence</i> setting for this instance.</p>
     * <p><b>Minimal Confidence</b> score is used as a confidence
     * boundary during the process of object detection. An object
     * that has been detected with a confidence score lower than
     * <i>minimalConfidence</i> is ignored. An object that has been
     * detected with a confidence score higher or equal than
     * <i>minimalConfidence</i> is added to the final result list.</p>
     *
     * @return This instance's minimalConfidence setting.
     */
    public float getMinimalConfidence() {
        return minimalConfidence;
    }

    /**
     * <p>Set the <i>minimalConfidence</i> setting for this instance.</p>
     * <p><b>Minimal Confidence</b> score is used as a confidence
     * boundary during the process of object detection. An object
     * that has been detected with a confidence score lower than
     * <i>minimalConfidence</i> is ignored. An object that has been
     * detected with a confidence score higher or equal than
     * <i>minimalConfidence</i> is added to the final result list.</p>
     *
     * @param minimalConfidence <i>minimalConfidence</i> for this instance
     * @throws IllegalStateException if the session has been closed already
     *                               by calling {@link #closeSession() closeSession()}
     */
    public void setMinimalConfidence(float minimalConfidence) throws IllegalStateException {
        checkSession();
        LOGGER.info("setting minimum confidence to: " + minimalConfidence);
        this.minimalConfidence = minimalConfidence;
    }

    /**
     * <p>Get this instance's debug mode setting.</p>
     * <p>If <b>debug mode</b> is enabled (set to true), the library
     * will open a {@link javax.swing.JFrame JFrame} for each processed
     * image with detections graphically highlighted.</p>
     *
     * @return debug mode setting for this instance
     */
    public boolean isDebugMode() {
        return debugMode;
    }

    /**
     * <p>Set this instance's debug mode setting.</p>
     * <p>If <b>debug mode</b> is enabled (set to true), the library
     * will open a {@link javax.swing.JFrame JFrame} for each processed
     * image with detections graphically highlighted.</p>
     *
     * @param debugMode turn the debug mode on or off
     * @throws IllegalStateException if the session has been closed already
     *                               by calling {@link #closeSession() closeSession()}
     */
    public void setDebugMode(boolean debugMode) throws IllegalStateException {
        checkSession();
        LOGGER.info("setting debug mode to " + debugMode);
        this.debugMode = debugMode;
    }

    /**
     * Closes the EggDetector session. This instance of EggDetector will not
     * be usable again.
     *
     * @throws IllegalStateException if the session has been closed already
     *                               by calling {@link #closeSession() closeSession()}
     */
    public void closeSession() throws IllegalStateException {
        checkSession();
        LOGGER.info("closing session");
        TensorFlowInferenceInterface.getInstance().close();
        sessionClosed = true;
    }

    /**
     * <p>Runs egg detection on a given <i>dir</i>.</p>
     *
     * @param dir a directory containing .jpg or .png files for object detection
     * @return {@link org.cvut.havluja1.eggdetector.SequenceClassifier}
     * @throws IllegalArgumentException if <i>dir</i> is not a directory or contains
     *                                  no images
     * @throws IllegalStateException    if the session has been closed already
     *                                  by calling {@link #closeSession() closeSession()}
     */
    public SequenceClassifier evaluate(File dir) throws IllegalArgumentException, IllegalStateException {
        checkSession();
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

    private void checkSession() throws IllegalStateException {
        if (sessionClosed) {
            throw new IllegalStateException("This session has been closed already.");
        }
    }

    @Override
    public String toString() {
        return "EggDetector{" +
                "minimalConfidence=" + minimalConfidence +
                ", debugMode=" + debugMode +
                '}';
    }
}
