package org.cvut.havluja1.tagger.model;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.cvut.havluja1.tagger.model.exceptions.NoMoreDataException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class FolderBrowserService implements IFolderBrowserService {

    private final String dataLocation;
    private final List<String> untaggedFolders;

    /**
     * Constructor.
     *
     * @param dataLocation Absolute path to a folder containing a structure of folders with raw data in them.
     */
    @Autowired
    public FolderBrowserService(@Value("${data.location}") String dataLocation) throws FileNotFoundException {
        if (dataLocation.substring(dataLocation.length() - 1).equals(File.separator)) {
            this.dataLocation = dataLocation;
        } else {
            this.dataLocation = dataLocation + File.separator;
        }

        if(!(new File(dataLocation)).exists()) {
            throw new FileNotFoundException("Provided directory does not exist or is inaccessible");
        }

        this.untaggedFolders = FolderScanner.scanFolder(this.dataLocation);
    }

    /**
     * Return another randomly selected folder with untagged data.
     *
     * @return Another random folder id.
     * @throws NoMoreDataException If there are no more folders in the list.
     */
    @Override
    public String getNextUntaggedFolder() throws NoMoreDataException {
        if (untaggedFolders.size() > 0) {
            return untaggedFolders.get(new Random().nextInt(untaggedFolders.size()));
        } else {
            throw new NoMoreDataException();
        }
    }

    /**
     * Return a list of absolutes paths of images in a given folder.
     *
     * @param folderId folder
     * @return All png file urls from given folder.
     * @throws FileNotFoundException Ff the folder's not found.
     */
    @Override
    public List<String> getFolderPictureList(String folderId) throws FileNotFoundException {
        File dir = new File(dataLocation + folderId);
        if (!dir.exists()) {
            throw new FileNotFoundException();
        }

        return Arrays.asList(dir.list((file, name) -> {
            File workingFile = new File(file.getAbsolutePath() + File.separator + name);
            return workingFile.isFile() && FilenameUtils.getExtension(name).equals("png");
        }));
    }
}
