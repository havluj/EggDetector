package org.cvut.havluja1.tagger;


import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class TaggerController {

    @RequestMapping("/")
    public String index(Model model) {
        model.addAttribute("name", "jan");
        return "index";
    }
}
