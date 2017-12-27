package org.cvut.havluja1.eggdetector;

import java.beans.XMLDecoder;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.cvut.havluja1.tagger.model.FolderData;
import org.cvut.havluja1.tagger.model.ImgData;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class SuccessRateTest {

    static final String DIR = "/run/media/jan/data/bachelor_thesis_data/data/egg_count_images";
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
    public void testSuccessRate() {
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
        int expectedLength = 0;
        int foundLength = 0;

        for (File dir : subdirs) {
            try {
                int foundTmp = evaluateDir(dir);
                int expectedTmp = getExpectedCount(dir);

                expectedLength += expectedTmp;
                foundLength += foundTmp;
                totalCnt++;
                if (foundTmp == expectedTmp) {
                    correctCnt++;
                }
                System.out.println("expected: " + expectedTmp + " | found: " + foundTmp);
            } catch (IOException | IllegalArgumentException e) {
            }
        }

        float cntSuccessRate = ((float) correctCnt) / ((float) totalCnt);
        float lengthSuccessRate = ((float) foundLength) / ((float) expectedLength);

        System.out.println("Found " + totalCnt + " directories.");
        System.out.println("EggDetector evaluated " + correctCnt + " directories correctly.");
        System.out.println(correctCnt + "/" + totalCnt + ": " + (cntSuccessRate * 100) + "% success rate.");
        System.out.println("Egg count of all folders added together: " + expectedLength + ".");
        System.out.println("EggDetector found " + foundLength + " eggs.");
        System.out.println(foundLength + "/" + expectedLength + ": " + (lengthSuccessRate * 100) + "% success rate.");
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
