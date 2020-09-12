import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class AutoCropCollage:
    def __init__(self):
        self.image = None
        self.tsh1 = 150
        self.tsh2 = 200
        self.vertical_size = 5
        self.horizontal_size =5
        self.vert_bound_ind_tshld = 10
        self.horiz_bound_ind_tshld = 10
        

    def read_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = np.array(image)
        f,ax = plt.subplots(figsize = (10,8))
        ax.imshow(self.image)
        ax.set_title('Original Collage')
    
    def detect_edges(self):
        edges = cv2.Canny(self.image[:,:,0], self.tsh1, self.tsh2)+ cv2.Canny(self.image[:,:,1], self.tsh1, self.tsh2)+  cv2.Canny(self.image[:,:,2], self.tsh1, self.tsh2)
        edges = cv2.Canny(edges, 0, 1)
        return edges
    
    def image_boundary_indicator(self):
        edges = self.detect_edges()
        
        #Filter out everything but the vertical lines from the edge detection result
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.vertical_size))
        mask_vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_structure)
        
        f,ax = plt.subplots(figsize = (10,8))
        ax.imshow(mask_vert)
        ax.set_title('Detected Vertical Edges')
        
        vert_bound_indicator = np.sum(mask_vert, axis = 0)
        
        #Filter out everything but the horizontal lines from the edge detection result
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (self.horizontal_size, 1))
        mask_horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_structure)
        
        f,ax = plt.subplots(figsize = (10,8))
        ax.imshow(mask_horiz)
        ax.set_title('Detected Horizontal Edges')
        
        horiz_bound_indicator = np.sum(mask_horiz, axis = 1)
        
        return vert_bound_indicator, horiz_bound_indicator
    
    def image_boundary_indices(self):
        vert_bound_ind, horiz_bound_ind = self.image_boundary_indicator()
        f,ax = plt.subplots()
        ax.plot(vert_bound_ind)
        ax.set_xlabel('Image Width Direction')
        ax.set_title('Vertical Boundary Indicator')
        peaks_vert, _ = find_peaks(vert_bound_ind, height = 100000, distance = self.image.shape[1]*0.1)
        
        
        peaks_horiz, _ = find_peaks(horiz_bound_ind, height = 100000, distance = self.image.shape[0]*0.1)
        f,ax = plt.subplots()
        ax.plot(horiz_bound_ind)
        ax.set_xlabel('Image height Direction')
        ax.set_title('Horizontal Boundary Indicator')
        
        return peaks_vert, peaks_horiz
    
    def split_collage(self, path):
        self.read_image(path)
        vert_bound_indcs, horiz_bound_indcs = self.image_boundary_indices()
        
        f,ax = plt.subplots(2,2, figsize = (10, 8))
        ax[0,0].imshow(self.image[0:horiz_bound_indcs[0], 0:vert_bound_indcs[0]])
        ax[0,1].imshow(self.image[0:horiz_bound_indcs[0], vert_bound_indcs[0]:])
        ax[1,0].imshow(self.image[horiz_bound_indcs[0]:, 0:vert_bound_indcs[0]])
        ax[1,1].imshow(self.image[horiz_bound_indcs[0]:, vert_bound_indcs[0]:])
        f.suptitle('Splitted Collage')