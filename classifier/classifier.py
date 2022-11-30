import logging
from datetime import datetime
import cv2 as cv
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans


class Classifier:

    def __init__(self, dataset, debug=False):
        self.logger = logging.getLogger(__name__)
        self.dataset = dataset
        self.debug = debug

    def train(self):
        x, y, model = self.get_data_and_labels(None, True)
        self.logger.info('Training new SVM.')
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(cv.ml.SVM_RBF)
        # svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        start = datetime.now()
        svm.trainAuto(x, cv.ml.ROW_SAMPLE, y)
        end = datetime.now()
        self.logger.info(f'Done training SVM in {end-start}.')
        filename = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        svm.save(f'{filename}-RBF-SVC.svm')
        self.logger.info(f'SVM saved to \'{filename}-RBF-SVC.svm\'.')
        return svm, model

    def test(self, svm, model):
        x, y, model = self.get_data_and_labels(model, False)
        _, result = svm.predict(x)
        mask = result == y
        correct = np.count_nonzero(mask)
        accuracy = (correct * 100.0 / result.size)
        self.logger.info(f'Finished with accuracy of {accuracy}%')
        return result, y

    def get_data_and_labels(self, model, isTrain):
        img_set = self.dataset.get_set()
        img_classes = self.dataset.get_classes()
        y = []
        x = None
        img_descs = []

        for class_number in range(len(img_set)):
            img_class = img_classes[class_number]
            self.logger.info(f'Computing ORB features for class \'{img_class}\' with size \'{len(img_set[class_number])}\'.')
            img_paths = img_set[class_number]

            for i in range(len(img_paths)):
                if self.debug:
                    self.logger.debug(f'Computing ORB features for image \'{str(img_paths[i])}\'.')
                img_pre = cv.imread(str(img_paths[i]))
                img = cv.cvtColor(img_pre, cv.COLOR_BGR2GRAY)
                desc, y = self.orb(img, img_descs, y, class_number)

        if isTrain:
            x, model = self.cluster(desc)
        else:
            x = self.img_to_vector(desc, model)
        y = np.int32(y)[:, np.newaxis]
        x = np.matrix(x, dtype=np.float32)
        return x, y, model
    
    def orb(self, img, img_descs, y, class_number):
        orb = cv.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            img_descs.append(des)
            y.append(class_number)
        return img_descs, y

    def sift(self, img, img_descs, y, class_number):
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            img_descs.append(des)
            y.append(class_number)
        return img_descs, y

    def cluster(self, img_descs):
        n_clusters = 64
        model = MiniBatchKMeans(n_clusters=64)
        training_descs = img_descs

        all_train_descs = [desc for desc_list in training_descs for desc in desc_list]
        all_train_descs = np.array(all_train_descs)

        if all_train_descs.shape[1] != 32:
            raise ValueError(f'Expected ORB descriptors to have 32 features. Got \'{int(all_train_descs.shape[1])}\'')

        model.fit(all_train_descs)
        self.logger.info('Done clustering. Begin generating BoVW histograms for each image.')

        img_clustered_words = [model.predict(raw_words) for raw_words in img_descs]

        img_bow_hist = np.array([np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

        x = img_bow_hist
        self.logger.info('Done generating BoVW histograms.')
        filename = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        with open(f'{filename}-K-Means.model', 'wb') as f:
            pickle.dump(model, f)
        self.logger.info(f'K-Means clustering modelsaved to \'{filename}-K-Means.model\'.')
        return x, model

    def img_to_vector(self, img_descs, model):
        clustered_descs = [model.predict(raw_words) for raw_words in img_descs]
        img_bow_hist = np.array([np.bincount(clustered_desc, minlength=model.n_clusters) for clustered_desc in clustered_descs])
        return img_bow_hist
