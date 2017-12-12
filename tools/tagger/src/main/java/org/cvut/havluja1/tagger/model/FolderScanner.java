package org.cvut.havluja1.tagger.model;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.io.FilenameUtils;

public class FolderScanner {

    /**
     * Creates a list of folder names in a given folder.
     *
     * @param location root folder location.
     * @return list of folder names (no paths)
     */
    public static ArrayList<String> scanFolder(String location) {
        File locFile = new File(location);

        if (!locFile.exists()) {
            return new ArrayList<>();
        }

        // example folder name: 20160430_073822_526_D
        final Pattern pattern = Pattern.compile("\\d{8}_\\d{6}_\\d{3}_D");
        List arr = Arrays.asList(locFile.list((File file, String name) -> {
            File workingDir = new File(file.getAbsolutePath() + File.separator + name);
            // needs to be a dir in a correct format
            if (!workingDir.isDirectory() && !pattern.matcher(name).matches()) {
                return false;
            }

            // if already tagged (contains imgdata.xml file)
            File imgDataFile = new File(workingDir, "imgdata.xml");
            if (imgDataFile.exists()) {
                return false;
            }

            // contains any pictures
            if (workingDir.list((f, n) -> {
                File workingFile = new File(f.getAbsolutePath() + File.separator + n);
                return workingFile.isFile() && FilenameUtils.getExtension(n).equals("png");
            }).length <= 0) {
                return false;
            }

            return true;
        }));

        return new ArrayList<>(arr);
    }
}
