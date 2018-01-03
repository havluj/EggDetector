package org.cvut.havluja1.libtest;

import java.io.File;

import org.cvut.havluja1.eggdetector.EggDetector;
import org.cvut.havluja1.eggdetector.SequenceClassifier;

public class Main {
    public static void main(String[] args) {
        EggDetector eggDetector = new EggDetector();
        SequenceClassifier sequenceClassifier = eggDetector.evaluate(new File(Main.class.getClassLoader().
                getResource("sample_images").getFile()));

        System.out.println("final count: " + sequenceClassifier.getFinalCount());
        System.out.println("individual scores: " + sequenceClassifier.getIndividualCounts());

        eggDetector.closeSession();
    }
}
