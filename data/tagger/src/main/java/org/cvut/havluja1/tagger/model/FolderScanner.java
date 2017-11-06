package org.cvut.havluja1.tagger.model;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class FolderScanner {
    public static List<String> scanFolder(String location) {
        File locFile = new File(location);

        if (!locFile.exists()) {
            return new ArrayList<>();
        }


        for (File dir : locFile.listFiles((file, name) -> {
            // is a dir
            // is in a correct format
            // contains any pictures
            return true;

            // if already tagged,
        })) {

        }
        return new ArrayList<>();
    }
}
