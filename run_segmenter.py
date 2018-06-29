import cv2 as cv
import sys, glob
from os.path import basename, dirname, realpath
from segmenter import Segmenter

if __name__ == "__main__":
    scriptpath = dirname(realpath(sys.argv[0]))
    files = []
    if len(sys.argv) == 1:
        imgs = glob.glob(scriptpath +"/raw/*/*.JPG")
    else:
        imgs = [realpath(sys.argv[1])]

    for img in imgs:
        obj = Segmenter(cv.imread(img))
        output = obj.run()
        cv.imwrite(scriptpath + "/processed/" + basename(img), output)
        msg = "writing file to " + scriptpath + "/processed/" + basename(img)
        print(msg)

# img = cv.imread("raw.JPG")
# obj = Segmenter(img)
# obj.run()

# cv.imwrite("test.JPG", obj.proc_img)
