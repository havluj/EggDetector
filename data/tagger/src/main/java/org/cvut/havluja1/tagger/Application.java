package org.cvut.havluja1.tagger;

import org.cvut.havluja1.tagger.model.IFolderBrowserService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @Bean
    public CommandLineRunner init(IFolderBrowserService iFolderBrowser) {
        return args -> {
//            iFolderBrowser.
        };
    }

}