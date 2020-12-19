# Bachelor thesis

This repository contains the text and the code of my bachelor thesis I wrote when studying Software Engineering at Czech Technical University in Prague. The objective of my thesis was to automate data collection for a research group studying bird behaviour, which I achieved by creating a software library that is able to automatically detect the number of eggs in a given video sequence. 

## Abstract

This thesis focuses on designing and creating a software library that is able to detect objects in images using neural networks.

In the first part of the thesis, goals and technical requirements are specified. The second part discusses the theory behind computer vision, neural networks, and TensorFlow. Using the technical specification and acquired theoretical knowledge, a detailed analysis of a possible solution is presented. There are two implementations of the analysis. The first implementation is developed using object detection, the second one using image recognition. In the last part of the thesis, a more effective implementation is chosen, which is then properly tested and verified.

The result of this thesis is a software library that is able to automatically detect the number of eggs in a given video sequence. The entire solution is written in Java and is easily intergratable with other parts of the BirdObserver project.

## Repository structure

```
| 
|--- bin ........................................ compiled binaries
|
|--- doc ........................................ javadoc
|
|--- text ....................................... source text of the thesis
|  |
|  |---thesis.pdf ............................... compiled text (pdf)
|
|--- src
|  |
|  |--- impl .................................... source code
|  |  |
|  |  |--- eggdetector .......................... source code of the main library
|  |  |
|  |  |--- egg_recognition ...................... image detection source code
|  |  |
|  |  |--- object-detection-training ............ object detection source code
|  |  |
|  |  |--- tools ................................ source code for other tools
|  |  |
|  |  |--- foldertrimmer ........................ source code of FolderTrimmer
|  |  |
|  |  |--- tagger ............................... source code of Tagger
|  |  |
|  |  |--- download.sh .......................... scripts for downloading training data
|  |  |
|  |  |--- libtest .............................. library integration testing 
|  | 
|  |--- thesis .................................. thesis text source code
|
|--- readme.txt ................................. repository structure
```
