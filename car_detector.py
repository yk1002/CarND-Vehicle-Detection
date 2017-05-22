#!/usr/bin/env python

from lesson_functions import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import os, sys, glob, time

class car_detector(object):

    def __init__(self):
        self.svc_ = None
        self.X_scaler_ = None
        self.color_space_ = None
        self.spatial_size_ = None
        self.hist_bins_ = None
        self.orient_ = None
        self.pix_per_cell_ = None
        self.cell_per_block_ = None
        self.hog_channel_ = None

    def build(self, car_image_list, notcar_image_list,
              color_space = 'YCrCb',
              spatial_size=(32, 32),
              hist_bins=32,
              orient=9,
              pix_per_cell=8,
              cell_per_block=2,
              hog_channel='ALL'):

        self.color_space_ = color_space
        self.spatial_size_ = spatial_size
        self.hist_bins_ = hist_bins
        self.orient_ = orient
        self.pix_per_cell_ = pix_per_cell
        self.cell_per_block_ = cell_per_block
        self.hog_channel_ = hog_channel

        # extract features from samples
        t_begin = time.time()
        car_features = extract_features(car_image_list,
                                        color_space=color_space, spatial_size=spatial_size,
                                        hist_bins=hist_bins, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                        spatial_feat=True, hist_feat=True, hog_feat=True)
        notcar_features = extract_features(notcar_image_list,
                                           color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                           spatial_feat=True, hist_feat=True, hog_feat=True)
        t_end = time.time()
        print(round(t_end - t_begin , 2), 'Seconds to extract HOG features...')

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

        # Fit a per-column scaler
        self.X_scaler_ = StandardScaler().fit(X)

        # Apply the scaler to X
        scaled_X = self.X_scaler_.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC 
        self.svc_ = LinearSVC()

        # Check the training time for the SVC
        begin_time = time.time()
        #params = { 'C': [0.1, 1, 10] }
        #svr = SVC()
        #self.svc_ = GridSearchCV(svr, params)
        self.svc_.fit(X_train, y_train)
        #print('Best params:', self.svc_.best_params_)
        end_time = time.time()
        print(round(end_time - begin_time, 2), 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc_.score(X_test, y_test), 4))

        # print('X_train[0].shape=', X_train[0].shape)
        # for i in range(20):
        #     pred = self.svc_.predict(X_train[i])
        #     print('case {}: real={}, pred={}'.format(i+1, pred, y_train[i]))

    def find_cars(self, img, ystart, yend, scale):
        assert self.svc_
        assert self.X_scaler_
        assert self.color_space_
        assert self.spatial_size_
        assert self.hist_bins_
        assert self.orient_
        assert self.pix_per_cell_
        assert self.cell_per_block_
        assert self.hog_channel_

        color_space = self.color_space_
        spatial_size = self.spatial_size_
        hist_bins = self.hist_bins_
        orient = self.orient_
        pix_per_cell = self.pix_per_cell_
        cell_per_block = self.cell_per_block_
        hog_channel = self.hog_channel_

        # training PNG images are scaled 0 to 1, but we expect img to be JPG, so scale here
        img = img.astype(np.float32)/255
    
        img_tosearch = img[ystart:yend,:,:]
        assert color_space == 'YCrCb'
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
    
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
        # Compute individual channel HOG features for the entire image
        assert hog_channel == 'ALL'
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        box_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler_.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.svc_.predict(test_features)
            
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    box = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart))
                    box_list.append(box)
                
        return box_list

    def draw_boxes(self, img, box_list, color=(0,0,255), width=6):
        draw_img = np.copy(img)
        for box in box_list:
            cv2.rectangle(draw_img, box[0], box[1], color, width)
        return draw_img

if __name__ == '__main__':

    if len(sys.argv) >= 3:
        # load existing detector from file
        pickle_file_path = sys.argv[2]
        with open(pickle_file_path, 'rb') as fd:
            detector = pickle.load(fd)
        print('Loaded detector from {}...'.format(pickle_file_path))
    else:
        # get car and non-car sample image paths
        cars = glob.glob('vehicles/**/*.png', recursive=True)
        notcars = glob.glob('non-vehicles/**/*.png', recursive=True)
        print('len(cars)={}'.format(len(cars)))
        print('len(notcars)={}'.format(len(notcars)))

        print('Building detector...')
        detector = car_detector()
        detector.build(cars, notcars)
        print('Done building detector')

        # save detector in file
        pickle_file_path = 'car_detector.p'
        with open(pickle_file_path, 'wb') as fd:
            pickle.dump(detector, fd)
        print('Saved detector to {}...'.format(pickle_file_path))

    # detect cars in given image
    test_image_path = sys.argv[1]
    print('Detecting cars in {}'.format(test_image_path))
    test_image = mpimg.imread(test_image_path)
#    ystart = int(test_image.shape[0]*0.4)
#    ystart = 0
    ystart = 400
    yend = 656
    print('ystart={}, yend={}'.format(ystart, yend))
    scale_list = [ 1.0, 1.5, 2.0 ]
#    scale_list = [ 0.75, 1.0, 1.25 ]
#    scale_list = [ 0.8, 1.0, 1.2 ]
    box_list = []
    for scale in scale_list:
        boxes = detector.find_cars(test_image, ystart, yend, scale)
        print('scale={}, len(boxes)={}'.format(scale, len(boxes)))
        box_list.extend(boxes)

    print('number of all boxes={}'.format(len(box_list)))
    image_with_boxes = detector.draw_boxes(test_image, box_list)

    plt.imshow(image_with_boxes)
    plt.show()
