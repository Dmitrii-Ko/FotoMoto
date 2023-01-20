import cv2


class Image:
    def __init__(self, name: str = ""):
        self.img = None
        self.name = name
        self.height, self.width, self.channel = 0, 0, 0

    def set_image(self, img):
        self.img = img
        self.__set_parameters()

    def open_image(self, path: str):
        self.img = cv2.imread(path)
        self.__set_parameters()

    def __set_parameters(self):
        self.height, self.width, self.channel = self.img.shape
