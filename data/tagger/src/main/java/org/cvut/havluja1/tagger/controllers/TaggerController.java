package org.cvut.havluja1.tagger.controllers;


import org.cvut.havluja1.tagger.model.NoMoreDataException;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class TaggerController {

    @RequestMapping("/")
    public String index(Model model) {
        try {
            return "redirect:/folder/" + getNextFolderId();
        } catch (NoMoreDataException e) {
            return "index";
        }
    }

    @RequestMapping("/folder/{folderId}")
    public String folder(@PathVariable("folderId") String folderId, Model model) {
        //        model.addAttribute("name", "jan");
        return "folder";
    }

    private String getNextFolderId() {
        // todo if there are no more folders: exception
//        throw new NoMoreDataException();
        return "";
    }
}
