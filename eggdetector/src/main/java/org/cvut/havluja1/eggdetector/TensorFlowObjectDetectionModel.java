package org.cvut.havluja1.eggdetector;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import java.util.logging.Logger;

import org.tensorflow.Graph;
import org.tensorflow.Operation;

class TensorFlowObjectDetectionModel implements Classifier {
    private static final Logger LOGGER = Logger.getLogger(TensorFlowObjectDetectionModel.class.getName());
    private static final int MAX_RESULTS = 100;
    private static final String INPUT_NAME = "image_tensor";
    private static final int INPUT_SIZE = 300; // nn is trained on 300x300 images
    private static final String LABEL = "egg";

    // Pre-allocated buffers.
    private int[] intValues;
    private byte[] byteValues;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface = TensorFlowInferenceInterface.getInstance();

    /**
     * Initializes a native TensorFlow session for classifying images.
     */
    public TensorFlowObjectDetectionModel() {
        final Graph g = inferenceInterface.graph();

        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(INPUT_NAME);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + INPUT_NAME + "'");
        }
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
        outputNames = new String[]{"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        intValues = new int[INPUT_SIZE * INPUT_SIZE];
        byteValues = new byte[INPUT_SIZE * INPUT_SIZE * 3];
        outputScores = new float[MAX_RESULTS];
        outputLocations = new float[MAX_RESULTS * 4];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
    }

    public List<Recognition> recognizeImage(final BufferedImage image) {
        // create a new, resized image
        BufferedImage thumbnail = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D tGraphics2D = thumbnail.createGraphics(); //create a graphics object to paint to
        tGraphics2D.setBackground(Color.WHITE);
        tGraphics2D.setPaint(Color.WHITE);
        tGraphics2D.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
        tGraphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        tGraphics2D.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE, null);

        // convert img to [INPUT_SIZE, INPUT_SIZE, 3]
        BufferedImage convertedImg = new BufferedImage(thumbnail.getWidth(), thumbnail.getHeight(), BufferedImage.TYPE_INT_RGB);
        convertedImg.getGraphics().drawImage(thumbnail, 0, 0, null);

        intValues = ((DataBufferInt) convertedImg.getRaster().getDataBuffer()).getData();

        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(INPUT_NAME, byteValues, 1, INPUT_SIZE, INPUT_SIZE, 3);

        // Run the inference call.
        inferenceInterface.run(outputNames, logStats);

        // Copy the output Tensor back into the output array.
        outputLocations = new float[MAX_RESULTS * 4];
        outputScores = new float[MAX_RESULTS];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);

        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        1,
                        (lhs, rhs) -> {
                            // Intentionally reversed to put high confidence at the head of the queue.
                            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * INPUT_SIZE,
                            outputLocations[4 * i] * INPUT_SIZE,
                            outputLocations[4 * i + 3] * INPUT_SIZE,
                            outputLocations[4 * i + 2] * INPUT_SIZE);
            pq.add(new Recognition(i, LABEL, outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
