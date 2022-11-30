import argparse
import logging
import cv2 as cv
import pickle
from pathlib import Path
from classifier import InputFile
from classifier import Classifier
from utils.log_formatter import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        dest='input_file',
                        required=True)
    parser.add_argument('--svm',
                        dest='svm_path',
                        type=lambda p: Path(p).absolute(),
                        help='path to saved SVM model')
    parser.add_argument('--model',
                        dest='model_path',
                        type=lambda p: Path(p).absolute(),
                        help='path to saved K-Means clustering model')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info('Logger initialized.')

    in_file_obj = InputFile(args.input_file)
    in_file_obj.generate_set()
    test_classifier = Classifier(in_file_obj, debug=False)
    svm = cv.ml.SVM_load(f'{str(args.svm_path)}')
    with open(f'{str(args.model_path)}', 'rb') as f:
        model = pickle.load(f)
    results, labels = test_classifier.test(svm, model)
    image_name_list = []
    with open(f'{args.input_file}') as f:
        for line in f:
            line = line.rstrip(line[-1])
            tkn = line.split("/")
            image_name = tkn[-1]
            image_name_list.append(image_name)
    class_dict = {0 : 'face', 1: 'car', 2: 'leaf', 3: 'motorcycle', 4: 'airplane'}
    f = open("output.txt", "w")
    i = 0
    for image in image_name_list:
        f.write(f'{image} {class_dict[int(results[i])]}\n')
        i += 1
    f.close()

if __name__ == '__main__':
    setup_logger()
    main()
