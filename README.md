# Yet Another Image Classifier
## Description
This project is an image classifier trained on classifying five categories of images: airplanes, faces, leaves, motorcycles, and cars.

### Dependencies
```
numpy==1.23.4
scikit_learn==1.1.3
opencv-python==4.6.0.66
```

### Training
The default location for training and testing images are `images/train/` and `images/test/` respectively. If another path is desired, the `--train-set` and `--test_set` flags can be used to specify these locations. Within these two directories there should be a directory for every class to be modeled. After training is completed, an SVM and a K-Means Clustering model will be saved to the project's root.
For example:
```
images/
├── test/
│   ├── airplane/
│   ├── car/
│   ├── face/
│   ├── leaf/
│   └── motorcycle/
└── train/
    ├── airplane/
    ├── car/
    ├── face/
    ├── leaf/
    └── motorcycle/
```
### Testing (with directory path)
Testing works the same as training. Default location is `images/test` but another can be specified with the `--test_set` flag. If no SVM or K-Means Clustering model are passed, then new models will be trained and then tested. To pass these models use the `--svm` and `--model` flags.
### Testing (with input text file)
Similarly, testing can be done by passing a text file of image filenames to be tested. This requires a special instance of the Dataset object, so `test.py` is used for filename testing. Pass in the SVM and model with the `--svm` and `--model` flags. After testing is complete, an `output.txt` will be saved to the project's root with a list of the image name and the classifier's predictions.

## Credits
The structure and a number of the helper methods in this project were developed by [kushalvyas](https://github.com/kushalvyas) as part of the [Bag of Visual Words](https://github.com/kushalvyas/Bag-of-Visual-Words-Python) project.

The five datasets used for training and testing were aggregated by the Caltech Vision Lab and can be found through the [Caltech Data](https://data.caltech.edu/) website. The datasets, along with their citations can be found linked on the [Caltech Vision Lab Datasets](https://www.vision.caltech.edu/datasets/) website.
