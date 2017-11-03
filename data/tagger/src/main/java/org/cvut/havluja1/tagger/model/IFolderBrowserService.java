package org.cvut.havluja1.tagger.model;

import java.io.FileNotFoundException;
import java.util.List;

public interface IFolderBrowserService {

    String getNextUntaggedFolder();

    List<String> getFolderPictureList(String folderId) throws FileNotFoundException;
}
