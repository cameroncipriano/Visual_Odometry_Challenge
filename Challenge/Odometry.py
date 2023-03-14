from glob import glob
import cv2, skimage, os
import numpy as np
import matplotlib.pyplot as plt

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
        
        self.frame_id = 0
        self.R_cam_pos = np.zeros((3, 3))
        self.t_cam_pos = np.zeros((3, 1))
        self.detector = cv2.ORB_create(nfeatures=8000, edgeThreshold=31, patchSize=31)
        self.tracking_feature_num = 0
        self.path = np.zeros((3,))
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    # empty list as default argument is INTENTIONAL because it acts as a store for each subsequent call
    def process_frame_odometry(self):
        print(f'Processing frame {self.frame_id}')

        if self.frame_id < 2:
            # Initialize two frames to perform two-frame odometry
            self.prev_frame = self.imread(self.frames[self.frame_id])
            self.curr_frame = self.imread(self.frames[self.frame_id + 1])

        else:
            self.prev_frame = self.curr_frame
            self.curr_frame = self.imread(self.frames[self.frame_id])

        # Calculate ORB Features from previous frame
        if self.tracking_feature_num < 3000:
            self.features = np.array([feature.pt for feature in self.detector.detect(self.prev_frame)], dtype=np.float32).reshape(-1, 1, 2)

        # Track features using KLT Tracker (Kanade-Lucas-Tomashi)
        self.next_features, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, self.curr_frame, self.features, None)
        
        self.retained_old =      self.features[status == 1]
        self.matched_new  = self.next_features[status == 1]

        # Copy is necessary to avoid modifying argument
        E, _ = cv2.findEssentialMat(self.matched_new, self.retained_old, self.focal_length, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, self.matched_new, self.retained_old, self.R_cam_pos.copy(), self.t_cam_pos.copy(), self.focal_length, self.pp, None)

        if self.frame_id < 2:
            self.R_cam_pos = R
            self.t_cam_pos = t
            self.frame_id += 2
        else:
            scale = self.get_scale(self.frame_id)
            self.t_cam_pos = self.t_cam_pos + (scale * (self.R_cam_pos @ t))
            self.R_cam_pos = self.R_cam_pos @ R
            self.frame_id += 1
        
        self.path = np.vstack((self.path, self.t_cam_pos.T))
        self.tracking_feature_num = self.matched_new.shape[0]


    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        while self.frame_id < len(self.frames):
            self.process_frame_odometry()
        
        return self.path

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path, path.shape)
    np.save('predictions.npy', path)

    with open('video_train/gt_sequence.txt', 'r') as f:
        data = [line.strip().split() for line in f.readlines()]
        gt_path = np.zeros((len(data), 3))

    gt_path[:, 0] = [float(pose[3]) for pose in data]
    gt_path[:, 1] = [float(pose[7]) for pose in data]
    gt_path[:, 2] = [float(pose[11]) for pose in data]

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1, projection='3d')
    ax1.plot3D(path[:, 0], path[:, 1], path[:, 2], 'r')
    ax1.plot3D(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], 'g')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(path[:, 0], path[:, 1], 'r')
    ax2.plot(gt_path[:, 0], gt_path[:, 1], 'g')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(path[:, 0], path[:, 2], 'r')
    ax3.plot(gt_path[:, 0], gt_path[:, 2], 'g')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(path[:, 1], path[:, 2], 'r')
    ax4.plot(gt_path[:, 1], gt_path[:, 2], 'g')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    plt.show()

    fig2 = plt.figure(2)
    frames = np.arange(801)
    ax12 = fig2.add_subplot(2,2,1)
    ax12.plot(frames, path[:, 0], 'r')
    ax12.plot(frames, gt_path[:, 0], 'g')
    ax12.set_xlabel('frame')
    ax12.set_ylabel('X')

    ax22 = fig2.add_subplot(2,2,2)
    ax22.plot(frames, path[:, 1], 'r')
    ax22.plot(frames, gt_path[:, 1], 'g')
    ax22.set_xlabel('frame')
    ax22.set_ylabel('Y')

    ax32 = fig2.add_subplot(2,2,3)
    ax32.plot(frames, path[:, 2], 'r')
    ax32.plot(frames, gt_path[:, 2], 'g')
    ax32.set_xlabel('frame')
    ax32.set_ylabel('Z')
    plt.show()
