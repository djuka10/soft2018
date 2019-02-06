#model koji predstavlja jednu cifru sa svojim koordinatama kontura, da li je presao liniju i njegov frame za neuronsku mrezu
class Digit:

    def __init__(self,x,y,w,h,passed,frame):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.passed = False
        self.frame = frame
