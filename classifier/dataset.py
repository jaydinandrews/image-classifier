import logging
import numpy as np


class Dataset:
    def __init__(self, path, debug=False):
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.path = path
        self.data_set = {}
        self.classes = []
        self.images = []
        self.counts = []

    def generate_set(self):
        categories = self.path.glob('*')
        i = 0
        for directory in categories:
            category = directory.name
            self.logger.info(f'Generating set from class \'{category}\'')
            self.classes.append(category)

            auxilary = []
            images = directory.glob('*.jpg')
            for image in images:
                if self.debug:
                    self.logger.debug(f'Reading file \'{str(image)}\'')
                self.images.append(image)
                auxilary.append(image)

            self.data_set[i] = auxilary
            i += 1

        self.classes = np.asarray(self.classes)
        self.logger.info(f"Dataset generated with size of {len(self.images)}")
        self.counts.append(0)

    def get_set(self):
        if len(self.images) == 0:
            self.generate_set()
        return self.data_set

    def get_classes(self):
        if len(self.classes) == 0:
            self.generate_set()
        return self.classes
