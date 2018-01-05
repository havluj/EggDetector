package org.cvut.havluja1.eggdetector;

import java.beans.XMLDecoder;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;

import org.cvut.havluja1.tagger.model.FolderData;
import org.cvut.havluja1.tagger.model.ImgData;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class SuccessRateTest {

    static final String DIR = "~/data/egg_count_images";
    static EggDetector eggDetector;

    @BeforeClass
    public static void setUp() {
        eggDetector = new EggDetector();
    }

    @AfterClass
    public static void close() {
        eggDetector.closeSession();
    }

    @Test
    public void testHighThreshold(){
        eggDetector.setMinimalConfidence(0.6f);
        System.out.println(eggDetector);
        testSuccessRate();
    }

    @Test
    public void testMiddleHighThreshold(){
        eggDetector.setMinimalConfidence(0.5f);
        System.out.println(eggDetector);
        testSuccessRate();
    }

    @Test
    public void testMiddleThreshold(){
        eggDetector.setMinimalConfidence(0.4f);
        System.out.println(eggDetector);
        testSuccessRate();
    }

    @Test
    public void testMiddleLowThreshold(){
        eggDetector.setMinimalConfidence(0.3f);
        System.out.println(eggDetector);
        testSuccessRate();
    }

    @Test
    public void testLowThreshold(){
        eggDetector.setMinimalConfidence(0.2f);
        System.out.println(eggDetector);
        testSuccessRate();
    }

    private void testSuccessRate() {
        File workingDir = new File(DIR);
        // get all directories
        File[] subdirs = workingDir.listFiles((file, name) -> {
            File tmp = new File(file, name);
            if (tmp.isDirectory()) {
                return true;
            }

            return false;
        });

        int totalCnt = 0;
        int correctCnt = 0;

        // distance
        int totalEggCount = 0;
        int lengthDifference = 0;

        for (File dir : subdirs) {
            try {
                int foundTmp = evaluateDir(dir);
                int expectedTmp = getExpectedCount(dir);

                totalEggCount += expectedTmp;
                totalCnt++;
                if (foundTmp == expectedTmp) {
                    correctCnt++;
                } else {
                    lengthDifference += Math.abs(foundTmp - expectedTmp);
                }
                System.out.println("expected: " + expectedTmp + " | found: " + foundTmp);
            } catch (IOException | IllegalArgumentException e) {
            }
        }

        float cntSuccessRate = ((float) correctCnt) / ((float) totalCnt);
        System.out.println("Found " + totalCnt + " directories.");
        System.out.println("EggDetector evaluated " + correctCnt + " directories correctly.");
        System.out.println(correctCnt + "/" + totalCnt + ": " + (cntSuccessRate * 100) + "% success rate.");
        System.out.println("Egg count of all folders added together: " + totalEggCount + ".");
        System.out.println("Distance (|real eggs - found eggs|): " + lengthDifference + " eggs (smaller is better).");
    }

    private int evaluateDir(File dir) throws IllegalArgumentException {
        return eggDetector.evaluate(dir).getFinalCount();
    }

    private int getExpectedCount(File dir) throws IOException {
        FolderData decodedSettings;
        try (FileInputStream fis = new FileInputStream(dir.getAbsolutePath() + "/imgdata.xml")) {
            XMLDecoder decoder = new XMLDecoder(fis);
            decodedSettings = (FolderData) decoder.readObject();
            decoder.close();
            fis.close();
        } catch (IOException e) {
            throw e;
        }

        int eggCount = 0;
        for (ImgData imgData : decodedSettings.getImgData()) {
            if (imgData.getEggCount() > eggCount) {
                eggCount = imgData.getEggCount();
            }
        }

        return eggCount;
    }
}
