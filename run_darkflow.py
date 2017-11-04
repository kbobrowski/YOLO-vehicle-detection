import sys, os, cv2
import numpy as np
sys.path.append('/darkflow')
from darkflow.net.build import TFNet
import namesgenerator
import binascii
from time import time
from skvideo.io import FFmpegWriter


weights = sys.argv[1]
video = sys.argv[2]
videoOut = sys.argv[3]

def get_corners(center, size):
    xmin = center[0] - size[1]/2
    ymin = center[1] - size[0]/2
    xmax = center[0] + size[1]/2
    ymax = center[1] + size[0]/2
    return xmin, ymin, xmax, ymax

def get_center_size(xmin, ymin, xmax, ymax):
    center = [(xmin + xmax)/2, (ymin + ymax)/2]
    size = [ymax - ymin, xmax-xmin]
    return center, size


class TrackedFrame(object):
    def __init__(self, center, img):
        self.img = img
        self.center = np.array(center)
        self.size = np.array(img.shape[:2]).astype(np.float64)
        self.centerSpeed = np.zeros(2)
        self.sizeSpeed = np.zeros(2)
        self.timeNotVisible = 0
        self.timeSinceCreation = 0
        self.hash = binascii.hexlify(os.urandom(16))
        self.name = namesgenerator.get_random_name()

    def update_speed(self, center, img):
        factorCenter = 0.1
        factorSize = 0.03
        centerChange = center - self.center
        sizeChange = np.array(img.shape[:2]) - self.size
        self.centerSpeed = factorCenter * centerChange
        self.sizeSpeed = factorSize * sizeChange
        self.timeNotVisible = 0

    def update(self, timeSpeed):
        self.center += self.centerSpeed
        if self.timeNotVisible == 0: self.size += self.sizeSpeed
        self.timeNotVisible += timeSpeed
        self.timeSinceCreation += 1

    def overlap(self, otherCenter, otherImg, scale='mean'):
        axmin, aymin, axmax, aymax = get_corners(otherCenter, otherImg.shape[:2])
        bxmin, bymin, bxmax, bymax = get_corners(self.center, self.size)
        dx = min(axmax, bxmax) - max(axmin, bxmin)
        dy = min(aymax, bymax) - max(aymin, bymin)
        aSize = otherImg.shape[0] * otherImg.shape[1]
        bSize = self.size[0] * self.size[1]
        meanSize = (aSize + bSize) / 2.
        if scale == 'mean': refSize = meanSize
        elif scale == 'self': refSize = bSize
        if dx >= 0  and dy >= 0:
            return dx * dy / refSize
        else:
            return 0


class Tracker(object):
    def __init__(self):
        self.frames = []

    def new_object(self, center, img):
        center = np.array(center)
        added = False
        bestOverlap = 0
        for frame in self.frames:
            overlap = frame.overlap(center, img)
            if overlap > bestOverlap:
                bestOverlap = overlap
                bestFrame = frame
        if bestOverlap > 0.1:
            bestFrame.update_speed(center, img)
        else:
            self.frames.append(TrackedFrame(center, img))

    def draw_frames(self, src):
        for i,frame in enumerate(self.frames):
            timeSpeed = 1
            otherFrames = [f for f in self.frames if f.hash is not frame.hash]
            for otherFrame in otherFrames:
                if frame.overlap(otherFrame.center, otherFrame.img, scale='self') > 0.5:
                    timeSpeed = 0.1
            frame.update(timeSpeed)
            if frame.timeSinceCreation > 500000./(frame.size[0]*frame.size[1]):
                xmin, ymin, xmax, ymax = get_corners(frame.center, frame.size)
                p1 = (int(xmin), int(ymin))
                p2 = (int(xmax), int(ymax))
                cv2.rectangle(src, p1, p2, (0,0,255), 5)
                pfont = (p1[0], p1[1]-20)
                cv2.putText(src, frame.name, pfont, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        self.frames = [frame for frame in self.frames if frame.timeNotVisible < 20]

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


if not os.path.isfile(weights):
    print("No such weights file: {}".format(weights))
    sys.exit(1)
if not os.path.isfile(video):
    print("No such input file: {}".format(video))
    sys.exit(1)
vidcap = cv2.VideoCapture(video)
fcount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
success, img = vidcap.read()
shape = img.shape
vidout = FFmpegWriter(videoOut)
if not success: sys.exit(1)
        
options = {'model': "/home/docker/yolo-vehicles.cfg", 'load': weights, 'gpu': 0.85, 'threshold': 0.3}

tfnet = TFNet(options)
i = 0
tracker = Tracker()
fps_avg = []



while True:
    i+=1
    t1 = time()
    pred = tfnet.return_predict(img)
    for p in pred:
        label = p['label']
        prob = p['confidence']
        topleft = p['topleft']
        bottomright = p['bottomright']
        xmin = topleft['x']
        ymin = topleft['y']
        xmax = bottomright['x']
        ymax = bottomright['y']
        width = xmax-xmin
        height = ymax-ymin
        if height < 20: continue
        if width < 20: continue
        centerx = (xmin + xmax)/2
        centery = (ymin + ymax)/2
        p1 = (int(xmin), int(ymin))
        p2 = (int(xmax), int(ymax))
        frameimg = img[ymin:ymin+height, xmin:xmin+width, :]
        tracker.new_object([centerx, centery], frameimg)
        cv2.rectangle(img, p1, p2, (0,255,255), 1)
    tracker.draw_frames(img)
    t2 = time()
    fps = 1/(t2-t1)
    fps_avg.append(fps)
    if len(fps_avg) > 200: fps_avg.pop(0)
    fps_print = int(np.mean(fps_avg))
    cv2.putText(img, "FPS: {}".format(fps_print), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    vidout.writeFrame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #cv2.imshow('img', img)
    #print("\rFrame: {}, FPS: {}".format(i, int(np.mean(fps_avg))), end='\r')
    printProgressBar(i, fcount, "Progress:", "({} FPS)".format(fps_print), length=20)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    success, img = vidcap.read()
    if not success: break

vidcap.release()
vidout.close()
cv2.destroyAllWindows()
