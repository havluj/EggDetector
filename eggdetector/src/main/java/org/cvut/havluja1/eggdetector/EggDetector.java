package org.cvut.havluja1.eggdetector;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.apache.commons.io.FilenameUtils;

public class EggDetector {

    private final static Logger LOGGER = Logger.getLogger(EggDetector.class.getName());

    public EggDetector() {
        LOGGER.info("feeding TensorFlow graph into memory...");
        TensorFlowInferenceInterface.getInstance();
        LOGGER.info("TensorFlow NN graph ready");
    }

    public SequenceClassifier evaluate(File dir) throws IllegalArgumentException {
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


        return new SequenceClassifier(imageList);
    }
}
