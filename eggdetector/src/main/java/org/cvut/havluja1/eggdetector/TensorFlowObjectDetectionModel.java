package org.cvut.havluja1.eggdetector;

import javax.imageio.ImageIO;
import javax.swing.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.logging.Logger;

public class TensorFlowObjectDetectionModel implements Classifier {
    private static final Logger LOGGER = Logger.getLogger(TensorFlowObjectDetectionModel.class.getName());

    private static final String LABEL = "egg";

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize) throws IOException {
        final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();

        InputStream labelsInput = null;
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        labelsInput = assetManager.open(actualFilename);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();


        d.inferenceInterface = new TensorFlowInferenceInterface(modelFilename);

        final Graph g = d.inferenceInterface.graph();

        d.inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(d.inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
        }
        d.inputSize = inputSize;
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
        d.outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        d.intValues = new int[d.inputSize * d.inputSize];
        d.byteValues = new byte[d.inputSize * d.inputSize * 3];
        d.outputScores = new float[MAX_RESULTS];
        d.outputLocations = new float[MAX_RESULTS * 4];
        d.outputClasses = new float[MAX_RESULTS];
        d.outputNumDetections = new float[1];
        return d;
    }

    private TensorFlowObjectDetectionAPIModel() {}

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        outputLocations = new float[MAX_RESULTS * 4];
        outputScores = new float[MAX_RESULTS];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);
        Trace.endSection();

        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        1,
                        (lhs, rhs) -> {
                            // Intentionally reversed to put high confidence at the head of the queue.
                            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * inputSize,
                            outputLocations[4 * i] * inputSize,
                            outputLocations[4 * i + 3] * inputSize,
                            outputLocations[4 * i + 2] * inputSize);
            pq.add(
                    new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}

//
//    private static final int BLOCK_SIZE = 32;
//    private static final int MAX_RESULTS = 3;
//    private static final int NUM_CLASSES = 20;
//    private static final int NUM_BOXES_PER_BLOCK = 5;
//    private static final int INPUT_SIZE = 416;
//    private static final String inputName = "input";
//    private static final String outputName = "output";
//
//    // Pre-allocated buffers.
//    private static int[] intValues;
//    private static float[] floatValues;
//    private static String[] outputNames;
//
//    // yolo 2
//    private static final double[] ANCHORS = {1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071};
//
//    // tiny yolo
//    //private static final double[] ANCHORS = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };
//
//    private static final String[] LABELS = {
//            "aeroplane",
//            "bicycle",
//            "bird",
//            "boat",
//            "bottle",
//            "bus",
//            "car",
//            "cat",
//            "chair",
//            "cow",
//            "diningtable",
//            "dog",
//            "horse",
//            "motorbike",
//            "person",
//            "pottedplant",
//            "sheep",
//            "sofa",
//            "train",
//            "tvmonitor"
//    };
//
//    private static TensorFlowInferenceInterface inferenceInterface;
//
//    public static void main(String[] args) {
//
//        //String modelDir = "/home/user/JavaProjects/TensorFlowJavaProject"; // Ubuntu
//        String modelAndTestImagesDir = "D:\\JavaProjects\\TensorFlowJavaProject"; // Windows
//        String imageFile = modelAndTestImagesDir + File.separator + "0.png"; // 416x416 test image
//
//        outputNames = outputName.split(",");
//        floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];
//
//        // yolo 2 voc
//        inferenceInterface = new TensorFlowInferenceInterface(Paths.get(modelAndTestImagesDir, "yolo-voc.pb"));
//
//        // tiny yolo voc
//        //inferenceInterface = new TensorFlowInferenceInterface(Paths.get(modelAndTestImagesDir, "graph-tiny-yolo-voc.pb"));
//
//        BufferedImage img;
//
//        try {
//            img = ImageIO.read(new File(imageFile));
//
//            BufferedImage convertedImg = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
//            convertedImg.getGraphics().drawImage(img, 0, 0, null);
//
//            intValues = ((DataBufferInt) convertedImg.getRaster().getDataBuffer()).getData();
//
//            List<Recognition> recognitions = recognizeImage();
//
//            System.out.println("Result length " + recognitions.size());
//
//            Graphics2D graphics = convertedImg.createGraphics();
//
//            for (Recognition recognition : recognitions) {
//                RectF rectF = recognition.getLocation();
//                System.out.println(recognition.getTitle() + " " + recognition.getConfidence() + ", " +
//                                           (int) rectF.left + " " + (int) rectF.top + " " + (int) rectF.right + " " + ((int) rectF.bottom));
//                Stroke stroke = graphics.getStroke();
//                graphics.setStroke(new BasicStroke(3));
//                graphics.setColor(Color.green);
//                graphics.drawRoundRect((int) rectF.left, (int) rectF.top, (int) rectF.width(), (int) rectF.height(), 5, 5);
//                graphics.setStroke(stroke);
//            }
//
//            graphics.dispose();
//            ImageIcon icon = new ImageIcon(convertedImg);
//            JFrame frame = new JFrame();
//            frame.setLayout(new FlowLayout());
//            frame.setSize(convertedImg.getWidth(), convertedImg.getHeight());
//            JLabel lbl = new JLabel();
//            frame.setTitle("Java (Win/Ubuntu), Tensorflow & Yolo");
//            lbl.setIcon(icon);
//            frame.add(lbl);
//            frame.setVisible(true);
//            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//
//    }
//
//    private static List<Classifier.Recognition> recognizeImage() {
//
//        for (int i = 0; i < intValues.length; ++i) {
//            floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
//            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
//            floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
//        }
//        inferenceInterface.feed(inputName, floatValues, 1, INPUT_SIZE, INPUT_SIZE, 3);
//
//        inferenceInterface.run(outputNames, false);
//
//        final int gridWidth = INPUT_SIZE / BLOCK_SIZE;
//        final int gridHeight = INPUT_SIZE / BLOCK_SIZE;
//
//        final float[] output = new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
//
//        inferenceInterface.fetch(outputNames[0], output);
//
//        // Find the best detections.
//        final PriorityQueue<Classifier.Recognition> pq =
//                new PriorityQueue<>(
//                        1,
//                        (lhs, rhs) -> {
//                            // Intentionally reversed to put high confidence at the head of the queue.
//                            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
//                        });
//
//        for (int y = 0; y < gridHeight; ++y) {
//            for (int x = 0; x < gridWidth; ++x) {
//                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
//                    final int offset =
//                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
//                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
//                                    + (NUM_CLASSES + 5) * b;
//
//                    final float xPos = (x + expit(output[offset + 0])) * BLOCK_SIZE;
//                    final float yPos = (y + expit(output[offset + 1])) * BLOCK_SIZE;
//
//                    final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * BLOCK_SIZE;
//                    final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) * BLOCK_SIZE;
//
//                    final RectF rect =
//                            new RectF(
//                                    Math.max(0, xPos - w / 2),
//                                    Math.max(0, yPos - h / 2),
//                                    Math.min(INPUT_SIZE - 1, xPos + w / 2),
//                                    Math.min(INPUT_SIZE - 1, yPos + h / 2));
//
//                    final float confidence = expit(output[offset + 4]);
//
//                    int detectedClass = -1;
//                    float maxClass = 0;
//
//                    final float[] classes = new float[NUM_CLASSES];
//                    for (int c = 0; c < NUM_CLASSES; ++c) {
//                        classes[c] = output[offset + 5 + c];
//                    }
//                    softmax(classes);
//
//                    for (int c = 0; c < NUM_CLASSES; ++c) {
//                        if (classes[c] > maxClass) {
//                            detectedClass = c;
//                            maxClass = classes[c];
//                        }
//                    }
//
//                    final float confidenceInClass = maxClass * confidence;
//                    if (confidenceInClass > 0.01) {
//                        pq.add(new Classifier.Recognition(detectedClass, LABELS[detectedClass], confidenceInClass, rect));
//                    }
//                }
//            }
//        }
//
//        final ArrayList<Recognition> recognitions = new ArrayList<>();
//        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
//            recognitions.add(pq.poll());
//        }
//        return recognitions;
//
//    }
//
//    private static float expit(final float x) {
//        return (float) (1. / (1. + Math.exp(-x)));
//    }
//
//    private static void softmax(final float[] vals) {
//        float max = Float.NEGATIVE_INFINITY;
//        for (final float val : vals) {
//            max = Math.max(max, val);
//        }
//        float sum = 0.0f;
//        for (int i = 0; i < vals.length; ++i) {
//            vals[i] = (float) Math.exp(vals[i] - max);
//            sum += vals[i];
//        }
//        for (int i = 0; i < vals.length; ++i) {
//            vals[i] = vals[i] / sum;
//        }
//    }
//
//    public void close() {
//        inferenceInterface.close();
//    }
//}
//
