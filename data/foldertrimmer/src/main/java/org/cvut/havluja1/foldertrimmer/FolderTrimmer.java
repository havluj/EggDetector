package org.cvut.havluja1.foldertrimmer;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

public class FolderTrimmer {
    public static void main(String[] args) throws IOException {
        File rootDir = new File(args[0]);

        if (!rootDir.exists() || !rootDir.isDirectory()) {
            throw new IllegalArgumentException("root dir does not exist");
        }

        System.out.println("finding useless data...");
        findAndDeleteEmptyDirs(rootDir);
        System.out.println("done");
    }

    private static void findAndDeleteEmptyDirs(File dir) {
        final boolean[] shouldBeDeleted = {true};
        final boolean leaveOnlyTaggedData = System.getProperty("leaveonlytagged").equalsIgnoreCase("true");

        File[] toBeProcessed = dir.listFiles((file, s) -> {
            File workingFile = new File(file, s);

            // if dir -> return true and tag this folder not to be deleted
            if (workingFile.isDirectory()) {
                shouldBeDeleted[0] = false;
                return true;
            }


            if (workingFile.isFile()) {
                if (leaveOnlyTaggedData) { // if we want to keep only tagged data
                    if (s.equals("imgdata.xml")) {
                        shouldBeDeleted[0] = false;
                        return false;
                    } else {
                        if (FilenameUtils.getExtension(s).equals("png")) {
                            return false;
                        }
                        return true;
                    }
                } else { // If file and is not xml, txt or png return true. If it is, tag this folder not to be deleted.
                    if (FilenameUtils.getExtension(s).equals("xml")
                            || FilenameUtils.getExtension(s).equals("png")
                            || FilenameUtils.getExtension(s).equals("txt")) {
                        shouldBeDeleted[0] = false;
                        return false;
                    } else {
                        return true;
                    }
                }
            }

            return true;
        });

        if (shouldBeDeleted[0]) {
            try {
                FileUtils.deleteDirectory(dir);
                System.out.println("[D] deleting dir: " + dir.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            if (toBeProcessed.length > 0) {
                for (File currFile : toBeProcessed) {
                    // if file -> delete
                    if (currFile.isFile()) {
                        if (currFile.delete()) {
                            System.out.println("[F] deleting file: " + currFile.getAbsolutePath());
                        }
                        continue;
                    }

                    // if dir -> recursive call
                    if (currFile.isDirectory()) {
                        findAndDeleteEmptyDirs(currFile);
                    }
                }
            }
        }
    }
}
