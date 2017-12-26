package org.cvut.havluja1.eggdetector;

import java.io.File;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class PackagedDataTest {

    static EggDetector eggDetector;
    static SequenceClassifier sequenceClassifier;

    @BeforeClass
    public static void setUp() {
        eggDetector = new EggDetector();
        sequenceClassifier = eggDetector.evaluate(new File(PackagedDataTest.class.getClassLoader().
                getResource("sample_images").getFile()));
    }

    @AfterClass
    public static void close() {
        eggDetector.closeSession();
    }

    @Test
    public void testIndividualScores() {
        Map<String, Integer> res = sequenceClassifier.getIndividualCounts();
        Assert.assertEquals((int) res.get("image1.png"), 9);
        Assert.assertEquals((int) res.get("image2.png"), 5);
        Assert.assertEquals((int) res.get("image3.jpg"), 3);
        Assert.assertEquals((int) res.get("image4.jpg"), 4);
        Assert.assertEquals((int) res.get("image5.jpg"), 3);
    }

    @Test
    public void testFinalScore() {
        // we expect 3, because we counted 3 eggs 2 times
        Assert.assertEquals((int) sequenceClassifier.getFinalCount(), 3);
    }
}


