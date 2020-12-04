import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import quaternion
from feature_tracker import tracker

gt_poses_path = 'kitti_data/gt_pose_01.npy'
pred_depths_path = 'kitti_data/pred_depth_01.npy'
pred_pose_path = 'kitti_data/pred_pose_01.npy'
image_path = 'kitti_data/01/{:010d}.jpg'


def get_depth(image_np, x, y):
    yy = int(y)
    xx = int(x)
    return image_np[yy, xx]


class PostSLAM:

    def __init__(self):
        self.WINDOW_SIZE = 10
        self.R_c_i = np.identity(3)
        self.R_i_c = np.identity(3)
        self.t_c_i = np.zeros(3)
        self.t_i_c = np.zeros(3)
        self.gt_poses_ENU = None
        self.pred_depths = None
        self.pred_rela_poses = None
        self.gt_Ps = []
        self.pred_poses = []

        # for optimization
        self.frame_ids = []
        self.states = np.array([])
        self.scales = np.array([])
        self.long_track_features = []


        self._load_data()

    def _load_data(self):
        """Load precomputed data."""
        # precomputed pose
        self.gt_poses_ENU = np.load(gt_poses_path, fix_imports=True, encoding='latin1')
        self.pred_depths = np.load(pred_depths_path, fix_imports=True, encoding='latin1')
        self.pred_rela_poses = np.load(pred_pose_path, fix_imports=True, encoding='latin1')
        # camera imu translation
        self.R_c_i = np.array([[ 9.98747206e-04, -9.99990382e-01,  4.25937849e-03],
                               [ 8.41690183e-03, -4.25082114e-03, -9.99955570e-01],
                               [ 9.99964049e-01,  1.03455328e-03,  8.41257521e-03]])
        self.t_c_i = np.array([[-0.25190787], [0.71945204], [-1.08908294]])
        self.R_i_c = self.R_c_i.T
        self.t_i_c = -self.R_i_c.dot(self.t_c_i).reshape(3, 1)

    def get_input_image(self, image_path, width, height):
        """Retrieve the image and resize it to w * h."""
        input_image = Image.open(image_path).convert('RGB')
        resized_input_image = input_image.resize((width, height), Image.LANCZOS)

        return resized_input_image

    def cam_ENU_transformation(self, pre_R_ENU, pre_t_ENU, R_c1_c0, t_c1_c0):
        """R_cam1_cam0,t_cam1_cam0: the first camera transform in the SECOND frame."""
        R_i0_i1 = self.R_i_c.dot(R_c1_c0.T).dot(self.R_c_i)
        cur_R_ENU = pre_R_ENU.dot(R_i0_i1)

        t_c0_c1 = -R_c1_c0.T.dot(t_c1_c0)
        cur_t_tmp = pre_R_ENU.dot(self.R_i_c.dot(t_c0_c1) + self.t_i_c - R_i0_i1.dot(self.t_i_c))
        cur_t_ENU = pre_t_ENU + cur_t_tmp

        return cur_R_ENU, cur_t_ENU

    def perform_slam(self, feature):
        """Do SLAM."""
        gt_R0_ENU = self.gt_poses_ENU[0][:3, :3]
        gt_P0_ENU = self.gt_poses_ENU[0][:3, 3:]
        pred_R_ENU = gt_R0_ENU
        pred_P_ENU = gt_P0_ENU
        self.gt_Ps.append(gt_P0_ENU)
        self.pred_poses.append(np.array(np.hstack((pred_R_ENU, pred_P_ENU))))

        scales = np.ones(self.WINDOW_SIZE)

        feed_width, feed_height = 640, 192
        feature.track_len = self.WINDOW_SIZE

        # track the first frame
        input_image = self.get_input_image(image_path.format(0), feed_width, feed_height)
        feature.track(np.asarray(input_image))
        self.frame_ids.append(0)

        for index in range(1, self.gt_poses_ENU.shape[0]):

            # get current ground truth pose
            gt_R_ENU = self.gt_poses_ENU[index][:3, :3]
            gt_P_ENU = self.gt_poses_ENU[index][:3, 3:]
            self.gt_Ps.append(gt_P_ENU)

            # get current predicted pose
            pred_R_c1_c0 = self.pred_rela_poses[index][:3, :3]
            pred_t_c1_c0 = self.pred_rela_poses[index][:3, 3].reshape(3, 1)

            pred_R_ENU, pred_P_ENU = self.cam_ENU_transformation(pred_R_ENU, pred_P_ENU,
                                                            pred_R_c1_c0, pred_t_c1_c0)
            self.pred_poses.append(np.array(np.hstack((pred_R_ENU, pred_P_ENU))))

            print('GT position {:d}'.format(index))
            print(gt_P_ENU)
            print('Pred position {:d}'.format(index))
            print(pred_P_ENU)
            print('...')

            # track current frame
            input_image = self.get_input_image(image_path.format(index), feed_width, feed_height)
            feature.track(np.asarray(input_image))

            # do optimization
            self.frame_ids.append(index)
            self.long_track_features = [i for i in feature.tracks if len(i) is len(feature.tracks[0])]
            self.optimize()

            # optimize after 10 frames
            if index > self.WINDOW_SIZE:
                self.frame_ids.pop(0)
                self.slide_window()

        """
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        ax1 = fig.add_subplot(2, 1, 1)  # 通过fig添加子图，参数：行数，列数，第几个。
        ax2 = fig.add_subplot(2, 1, 2)  # 通过fig添加子图，参数：行数，列数，第几个。
    
        plot1 = ax1.plot(np.array(gt_Ps)[:, 0, 0], np.array(gt_Ps)[:, 1, 0], linestyle='--', alpha=0.5, color='r',
                         label='legend1')  # 线图：linestyle线性，alpha透明度，color颜色，label图例文本
        plot2 = ax2.plot(np.array(pred_Ps)[:, 0, 0], np.array(pred_Ps)[:, 1, 0], linestyle='--', alpha=0.5, color='r',
                         label='legend2')  # 线图：linestyle线性，alpha透明度，color颜色，label图例文本
    
        plt.show()
        print('done')
        """

    def optimize(self):
        """Nonlinear optimization."""
        # get state parameters for optimization
        self.get_state()


        self.set_state()

        pass

    def slide_window(self):
        pass

    def get_state(self):
        self.states = np.array([])
        dim = len(self.frame_ids) * (8 + len(self.long_track_features))
        self.states.resize((dim, 1))

        index = 0
        frame_count = 0
        for frame_id in self.frame_ids:
            # position 3x1
            self.states[index:(index + 3)] = self.pred_poses[frame_id][:, [3]]
            index += 3
            # orientation as quaternion 4x1
            q = quaternion.from_rotation_matrix(self.pred_poses[frame_id][:, :3])
            self.states[index:(index + 4)] = quaternion.as_float_array(q).reshape(4, 1)
            index += 4
            # scale 1x1
            self.states[index] = 1.0
            index += 1
            for feature_pts in self.long_track_features:
                # depth value at feature coordinates 1x1
                pt_x = feature_pts[frame_count][0]
                pt_y = feature_pts[frame_count][1]
                depth = get_depth(self.pred_depths[frame_id], pt_x, pt_y)
                self.states[index] = depth
                index += 1
            frame_count += 1

    def set_state(self):
        self.states = np.array([])
        dim = len(self.frame_ids) * (8 + len(self.long_track_features))
        self.states.resize((dim, 1))

        index = 0
        frame_count = 0
        for frame_id in self.frame_ids:
            # position 3x1
            self.pred_poses[frame_id][:, [3]] = self.states[index:(index + 3)]
            index += 3
            # orientation as rotation matrix 3x3
            q = quaternion.as_quat_array(self.states[index:(index + 4)])
            R = quaternion.as_rotation_matrix(q)
            self.pred_poses[frame_id][:, :3] = R
            index += 4
            # scale 1x1
            # first save scale value
            scale = self.states[index]
            index += 1
            for feature_pts in self.long_track_features:
                # depth value at feature coordinates 1x1
                pt_x = int(feature_pts[frame_count][0])
                pt_y = int(feature_pts[frame_count][1])
                depth = self.states[index]
                self.pred_depths[frame_id][pt_y, pt_x] = depth
                index += 1
            # multiply all depth by scale
            self.pred_depths[frame_id] *= scale
            frame_count += 1

    def compute_jacobi(self):
        pass


if __name__ == '__main__':

    feature = tracker()
    post_slam = PostSLAM()
    post_slam.perform_slam(feature)

