package org.cvut.havluja1.eggdetector;

import java.io.File;

public class Test {
    public static void main(String[] args) throws Exception {
        EggDetector eggDetector = new EggDetector();
        SequenceClassifier seq = eggDetector.evaluate(new File("/run/media/jan/data/bachelor_thesis_data/data/egg_count_images/20160420_165438_913_D"));

        System.out.println(seq.getFinalCount());
    }
}


