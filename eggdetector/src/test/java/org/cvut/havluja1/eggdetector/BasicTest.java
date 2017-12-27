package org.cvut.havluja1.eggdetector;

import java.io.File;

import org.junit.Test;

public class BasicTest {

    private static final String DIR = "/run/media/jan/data/bachelor_thesis_data/data/egg_count_images/20160421_191543_725_D/";

    @Test
    public void test() {
        EggDetector eggDetector = new EggDetector();
        eggDetector.setDebugMode(true);
        SequenceClassifier seq = eggDetector.evaluate(new File(DIR));
        System.out.println("Final result: " + seq.getFinalCount());
        System.out.println("Individual scores: " + seq.getIndividualCounts());
        eggDetector.closeSession();
        try {
            Thread.sleep(1000 * 60 * 2); // sleep for 2 min to view the GUI
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
