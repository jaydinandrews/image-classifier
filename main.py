import argparse
import logging
import sys
import cv2 as cv
import pickle
from pathlib import Path
from classifier import Dataset
from classifier import Classifier
from utils.log_formatter import setup_logger


def main():
    description = ("Image classifier utilizing Oriented Features from "
                   "Accelerated Segment Test and Rotated Binary Robust "
                   "Independent Elementary Features Descriptor Feature "
                   "Extraction and Support Vector Machine Supervised "
                   "Learning Models - say that ten times fast ;)")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--action',
                        dest='action',
                        choices=['train', 'test'],
                        required=True,
                        help='directory of training images')
    parser.add_argument('--svm',
                        dest='svm_path',
                        type=lambda p: Path(p).absolute(),
                        help='path to saved SVM model')
    parser.add_argument('--model',
                        dest='model_path',
                        type=lambda p: Path(p).absolute(),
                        help='path to saved K-Means clustering model')
    parser.add_argument('--train-path',
                        dest='train_path',
                        type=lambda p: Path(p).absolute(),
                        default=Path('./images/train/').absolute(),
                        help='directory of training images')
    parser.add_argument('--test-path',
                        dest='test_path',
                        type=lambda p: Path(p).absolute(),
                        default=Path('./images/test/').absolute(),
                        help='directory of testing images')
    parser.add_argument('--debug',
                        '-d',
                        action='store_true',
                        default=False,
                        help='increase output verbosity for debugging')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info('Logger initialized.')

    if args.action == 'train':
        train_dataset = Dataset(args.train_path, debug=args.debug)
        train_dataset.generate_set()
        train_classifier = Classifier(train_dataset, debug=args.debug)
        svm, model = train_classifier.train()
    else: # args.action == 'test':
        test_dataset = Dataset(args.test_path, debug=args.debug)
        test_dataset.generate_set()
        if args.svm_path or args.model_path:
            try:
                test_classifier = Classifier(test_dataset, debug=args.debug)
                try:
                    svm = cv.ml.SVM_load(f'{str(args.svm_path)}')
                except Exception:
                    logger.error('Failed to load saved SVM model from \
                            \'{str(args.svm_path)}\'')
                    sys.exit(1)
                try:
                    with open(f'{str(args.model_path)}', 'rb') as f:
                        model = pickle.load(f)
                except Exception:
                    logger.error('Failed to load saved K-Means clustering \
                            model from \'{str(args.model_path)}\'')
                    sys.exit(1)
                results, labels = test_classifier.test(svm, model)
            except Exception:
                logger.warning('Training new SVM.')
                train_new(args)
        else:
            logger.warning('Training new SVM.')
            train_new(args)

def train_new(args):
    train_dataset = Dataset(args.train_path, debug=args.debug)
    train_dataset.generate_set()
    test_dataset = Dataset(args.test_path, debug=args.debug)
    test_dataset.generate_set()

    train_classifier = Classifier(train_dataset, debug=args.debug)
    svm, model = train_classifier.train()
    test_classifier = Classifier(test_dataset, debug=args.debug)
    results, labels = test_classifier.test(svm, model)


if __name__ == '__main__':
    setup_logger()
    main()
