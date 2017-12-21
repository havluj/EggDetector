package org.cvut.havluja1.eggdetector;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;

public class EggDetector {

    private final static Logger LOGGER = Logger.getLogger(EggDetector.class.getName());
    private final static String FROZEN_GRAPH = "frozen_inference_graph.pb";

    private final Graph graph;

    public EggDetector() {
        Graph graph = new Graph();
        try {
            graph.importGraphDef(IOUtils.toByteArray(getClass().getClassLoader().getResourceAsStream(FROZEN_GRAPH)));
        } catch (IOException e) {
            System.err.println("Failed to read [" + FROZEN_GRAPH + "]: " + e.getMessage());
            System.exit(1);
        }
        this.graph = graph;
        LOGGER.info("Loaded Tensorflow model");
    }

    public SequenceClassifier evaluate(File dir) throws IllegalArgumentException {
        if (!dir.isDirectory()) {
            throw new IllegalArgumentException(dir.getAbsolutePath() + " is not a directory.");
        }

        List imageList = Arrays.asList(dir.list((File file, String name) -> {
            File workingFile = new File(file.getAbsolutePath() + File.separator + name);
            return workingFile.isFile()
                    && (FilenameUtils.getExtension(name).equals("png")
                    || FilenameUtils.getExtension(name).equals("jpg"));
        }));

        if (imageList.isEmpty()) {
            throw new IllegalArgumentException(dir.getAbsolutePath() + " does not contain any pictures.");
        }


        return new SequenceClassifier(imageList, graph);
    }

//    public SequenceClassifier evaluate(String apacheDirListing) {
//        return new SequenceClassifier(new ArrayList<>(), graph);
//    }
}
