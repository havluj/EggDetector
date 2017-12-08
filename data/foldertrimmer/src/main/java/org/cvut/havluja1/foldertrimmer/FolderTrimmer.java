package org.cvut.havluja1.foldertrimmer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

public class FolderTrimmer {
    public static void main(String[] args) throws IOException {
        File rootDir = new File(args[0]);

        if (!rootDir.exists() || rootDir.isDirectory()) {
            throw new IllegalArgumentException("root dir does not exist");
        }

        List<String> emptyDirs = findAndDeleteEmptyDirs(rootDir);
        emptyDirs.forEach(x -> System.out.println(x));
    }

    private static List<String> findAndDeleteEmptyDirs(File dir) {
        List<String> deletedFolders = new ArrayList<String>();
        boolean shouldBeDeleted = false;

        File[] toBeProcessed = dir.listFiles();
        // list everything
        // if file and is not xml, txt or png return, if it is, tag this folder not to be deleted
        // if dir -> return and tag this folder not to be deleted

        if(shouldBeDeleted) {
            try {
                FileUtils.deleteDirectory(dir);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            // foreach on all results
            // if dir -> recursive call
            // if file -> delete
        }

        return deletedFolders;
    }
}
