package org.cvut.havluja1.tagger;

import java.io.FileNotFoundException;
import java.util.List;

import org.cvut.havluja1.tagger.model.IFolderBrowserService;
import org.cvut.havluja1.tagger.model.exceptions.NoMoreDataException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class TaggerController {

    final IFolderBrowserService folderBrowser;

    @Autowired
    public TaggerController(IFolderBrowserService folderBrowser) {
        this.folderBrowser = folderBrowser;
    }

    @RequestMapping("/")
    public String index() {
        try {
            return "redirect:/folder/" + getNextFolderId();
        } catch (NoMoreDataException e) {
            return "index";
        }
    }

    @RequestMapping("/folder/{folderId}")
    public String folder(Model model, @PathVariable("folderId") String folderId) {
        List<String> pics;
        try {
            pics = folderBrowser.getFolderPictureList(folderId);
        } catch (FileNotFoundException e) {
            return "redirect:/foldernotfound/" + folderId;
        }

        model.addAttribute("folderId", folderId);
        model.addAttribute("pics", pics);
        return "folder";
    }

    @RequestMapping("/foldernotfound/{folderId}")
    public String folderNotFound(@PathVariable("folderId") String folderId, Model model) {
        model.addAttribute("folderId", folderId);

        return "foldernotfound";
    }

    private String getNextFolderId() throws NoMoreDataException {
        return folderBrowser.getNextUntaggedFolder();
    }
}
