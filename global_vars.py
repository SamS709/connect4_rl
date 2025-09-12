# global variables that we can access in other files

LIGHT_GREEN = [169 / 256, 221 / 256, 175 / 256, 1]
GREEN = [62 / 256, 182 / 256, 75 / 256, 1]
DARK_GREEN = [16 / 256, 118 / 256, 0, 1]
LIGHT_RED = [256/256,187/256,187/256,1]
RED = [237/256,79/256,79/256,1]
DARK_RED = [170/256,14/256,14/256,1]
LIGHT_BLUE = [182 / 256, 229 / 265, 246 / 256, 1]
BLUE = [112 / 256, 159 / 265, 256 / 256, 1]
DARK_BLUE = [82 / 256, 129 / 265, 256 / 256, 1]
SAND = [219/256,195/256,151/256,1]
WHITE = [1, 1, 1, 1]
BLACK = [0, 0, 0, 1]
TRANSPARENT = [1, 1, 1, 0]
SEMI_TRANSPARENT = [1, 1, 1, 0.3]
MAROON = [115/256, 63/256, 11/256, 1]

ROBOT = False

MODEL_NAME = ""

class var:
    def __init__(self):
        self.MODE = ""
        self.LEVEL = ""
        self.GAME = ""
        self.model_name = ""

    def __str__(self):
        return f"Mode: {self.MODE}, Level: {self.LEVEL}, Game: {self.GAME}, AI Model: {self.model_name}"

var1 = var()
