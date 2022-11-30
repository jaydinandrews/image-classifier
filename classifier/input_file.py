import logging
import numpy as np
from pathlib import Path


class InputFile:
    def __init__(self, filename, debug=False):
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.filename = filename
        self.data_set = {}
        self.classes = []
        self.images = []
        self.counts = []

    def generate_set(self):
        last_class = ''
        i = 1
        j = 0
        auxilary = []
        with open(self.filename) as f:
            for line in f:
                line = line.rstrip(line[-1])
                tkn = line.split("/")
                image_name = tkn[-1].rstrip(tkn[-1][-1])
                class_name = tkn[-2]
                self.images.append(Path(line).absolute())
                auxilary.append(Path(line).absolute())
                if(i % 100 == 0 and i != 0):
                    self.classes.append(class_name)
                    self.data_set[j] = auxilary
                    auxilary = []
                    j += 1
                i += 1
        self.classes = np.asarray(self.classes)
        self.logger.info(f'InputFile Dataset generated with size of {len(self.images)}')
        self.counts.append(0)

    def get_set(self):
        if len(self.images) == 0:
            self.generate_set()
        return self.data_set

    def get_classes(self):
        if len(self.classes) == 0:
            self.generate_set()
        return self.classes
