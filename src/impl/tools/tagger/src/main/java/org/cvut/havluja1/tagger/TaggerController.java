package org.cvut.havluja1.tagger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.cvut.havluja1.tagger.model.FolderData;
import org.cvut.havluja1.tagger.model.IFolderBrowserService;
import org.cvut.havluja1.tagger.model.ImgData;
import org.cvut.havluja1.tagger.model.exceptions.NoMoreDataException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
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
            return "redirect:/folder/" + folderBrowser.getNextUntaggedFolder();
        } catch (NoMoreDataException e) {
            return "index";
        }
    }

    @GetMapping("/folder/{folderId}")
    public String folder(Model model, @PathVariable("folderId") String folderId) {
        List<String> pics;
        try {
            pics = folderBrowser.getFolderPictureList(folderId);
        } catch (FileNotFoundException e) {
            return "redirect:/foldernotfound/" + folderId;
        }

        List<ImgData> imgs = new ArrayList<>();
        for (String pic : pics) {
            ImgData workingImg = new ImgData();
            workingImg.setName(pic);
            imgs.add(workingImg);
        }
        FolderData workingData = new FolderData();
        workingData.setImgData(imgs);

        model.addAttribute("folderId", folderId);
        model.addAttribute("folderData", workingData);
        return "folder";
    }

    @PostMapping("/folder/{folderId}")
    public String folderSubmit(Model model, @PathVariable("folderId") String folderId,
                               @ModelAttribute FolderData folderData) {
        try {
            folderBrowser.writeData(folderId, folderData);
            folderBrowser.removeFromList(folderId);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return index();
    }

    @RequestMapping("/foldernotfound/{folderId}")
    public String folderNotFound(@PathVariable("folderId") String folderId, Model model) {
        model.addAttribute("folderId", folderId);

        return "foldernotfound";
    }
}
