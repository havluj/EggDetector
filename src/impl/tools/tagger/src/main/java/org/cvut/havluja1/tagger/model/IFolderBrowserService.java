package org.cvut.havluja1.tagger.model;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public interface IFolderBrowserService {

    String getNextUntaggedFolder();

    List<String> getFolderPictureList(String folderId) throws FileNotFoundException;

    void writeData(String folderId, FolderData folderData) throws IOException;

    void removeFromList(String folderId);
}
