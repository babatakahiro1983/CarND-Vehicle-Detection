import numpy as np
import cv2
import tkinter.filedialog as tkfd
from lesson_functions import *
from scipy.ndimage.measurements import label
from sklearn.externals import joblib


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              window, cells_per_step):
    test_features = []
    box = []

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1), y=None, copy=None)
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                box.append([(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)])

    return draw_img, box


class ObjectDetector(object):
    def __init__(self):
        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = joblib.load("vehicle_detector_tmp.pkl")
        self.box_tmp_1 = []
        self.box_tmp_2 = []
        self.box_tmp_3 = []
        self.box_tmp_4 = []
        self.box_tmp_5 = []
        self.box_tmp = []
        self.boxes = []
        self.heat = []
        self.svc = svc
        self.X_scaler = X_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins

    def process_image(self, image):

        # while (cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret == True:

        #######################################################################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ystart = 400
        ystop = 656

        scale = 1.5
        window = 64
        cells_per_step = 1
        out_img, boxes_1 = find_cars(image, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block,
                                     self.spatial_size, self.hist_bins, window, cells_per_step)

        scale = 1
        window = 64
        cells_per_step = 1
        out_img, boxes_2 = find_cars(image, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block,
                                     self.spatial_size, self.hist_bins, window, cells_per_step)

        scale = 2
        window = 128
        cells_per_step = 1
        out_img, boxes_3 = find_cars(image, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block,
                                     self.spatial_size, self.hist_bins, window, cells_per_step)

        self.boxes = boxes_1 + boxes_2 + boxes_3

        #######################################################################################################

        # Read in a pickle file with bboxes saved
        # Each item in the "all_bboxes" list will contain a
        # list of boxes for one of the images shown above
        box_list = self.boxes
        self.box_tmp_5 = self.box_tmp_4
        self.box_tmp_4 = self.box_tmp_3
        self.box_tmp_3 = self.box_tmp_2
        self.box_tmp_2 = self.box_tmp_1
        self.box_tmp_1 = box_list
        box_tmp = self.box_tmp_5 + self.box_tmp_4 + self.box_tmp_3 + self.box_tmp_2 + self.box_tmp_1

        # Read in image similar to one shown above
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, box_tmp)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 10)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        #######################################################################################################
        showed_image = draw_img
        cv2.imwrite("./video_images.jpg", showed_image)
        # showed_image = image
        #
        # # Show movie
        # # out.write(showed_image)
        # # cv2.imshow('frame', showed_image)

        return cv2.cvtColor(showed_image, cv2.COLOR_BGR2RGB)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            # else:
            #     break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()