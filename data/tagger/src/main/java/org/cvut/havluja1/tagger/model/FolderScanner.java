package org.cvut.havluja1.tagger.model;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.io.FilenameUtils;

public class FolderScanner {
    public static List<String> scanFolder(String location) {
        File locFile = new File(location);

        if (!locFile.exists()) {
            return new ArrayList<>();
        }

        List<String> result = new ArrayList<>();

        // example folder name: 20160430_073822_526_D
        final Pattern pattern = Pattern.compile("\\d{8}_\\d{6}_\\d{3}_D");
        final File[] fileList = locFile.listFiles((File file, String name) -> {
            // needs to be a dir in a correct format
            if (!file.isDirectory() && !pattern.matcher(name).matches()) {
                return false;
            }

            // if already tagged
            // todo check the db

            // contains any pictures
            if (file.list((f, n) -> FilenameUtils.getExtension(n).equals("png")).length <= 0) {
                return false;
            }

            return true;
        });
        for (File dir : fileList) {
            result.add(dir.getAbsolutePath());
        }

        return result;
    }
}
