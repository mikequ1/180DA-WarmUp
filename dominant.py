# REFERENCES
# - Dominant color: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# - Overlaying: https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
#
# IMPROVEMENTS
# I used the dominant color tutorial and then added a bar containing the three most dominant colors on the 
# top left corner of the same frame.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come her
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    display = frame
    frame = frame.reshape((frame.shape[0] * frame.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(frame)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    # overlay color bar
    x_offset=y_offset=50
    display[y_offset:y_offset+bar.shape[0], x_offset:x_offset+bar.shape[1]] = bar
    cv2.imshow('frame',display)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()