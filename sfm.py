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
from layers import Project3D, BackprojectDepth

gt_poses_path = 'kitti_data/gt_pose_01.npy'
pred_depths_path = 'kitti_data/pred_depth_01.npy'
pred_pose_path = 'kitti_data/pred_pose_01.npy'
image_path = 'kitti_data/01/{:010d}.jpg'


def get_depth(image_np, x, y):
    yy = int(y)
    xx = int(x)
    return image_np[yy, xx]


def vec2skew(v):
    """Convert vector to skew matrix."""
    skew = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return skew


class PostSLAM:
    def __init__(self):
        self.WINDOW_SIZE = 10
        self.R_c_i = np.identity(3)
        self.R_i_c = np.identity(3)
        self.t_c_i = np.zeros(3)
        self.t_i_c = np.zeros(3)
        self.K_cam = np.identity(3)
        self.gt_poses_ENU = None
        self.pred_depths = None
        self.pred_rela_poses = None
        self.gt_Ps = []
        self.pred_poses = []
        self.feed_width = 640
        self.feed_height = 192

        # for optimization
        self.measure_count = 0
        self.frame_ids = []
        self.states = np.array([])
        self.long_track_features = []
        self.jacobians = np.array([])
        self.residuals = np.array([])
        self.frameid2state = {}
        self._load_data()

    def _load_data(self):
        """Load precomputed data."""
        # precomputed pose
        self.gt_poses_ENU = np.load(gt_poses_path, fix_imports=True, encoding='latin1')
        self.pred_depths = np.load(pred_depths_path, fix_imports=True, encoding='latin1')
        self.pred_rela_poses = np.load(pred_pose_path, fix_imports=True, encoding='latin1')
        # camera imu translation
        self.R_c_i = np.array([[9.98747206e-04, -9.99990382e-01,  4.25937849e-03],
                               [8.41690183e-03, -4.25082114e-03, -9.99955570e-01],
                               [9.99964049e-01,  1.03455328e-03,  8.41257521e-03]])
        self.t_c_i = np.array([[-0.25190787], [0.71945204], [-1.08908294]])
        self.R_i_c = self.R_c_i.T
        self.t_i_c = -self.R_i_c.dot(self.t_c_i).reshape(3, 1)
        self.K = np.array([[0.58, 0, 0.5],
                           [0, 1.92, 0.5],
                           [0, 0, 1]], dtype=np.float)
        self.K[0, :] *= self.feed_width
        self.K[1, :] *= self.feed_height
        self.inv_K = np.linalg.pinv(self.K)

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

        # scales = np.ones(self.WINDOW_SIZE)

        feature.track_len = self.WINDOW_SIZE

        # track the first frame
        input_image = self.get_input_image(image_path.format(0), self.feed_width, self.feed_height)
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

            #print('GT position {:d}'.format(index))
            #print(gt_P_ENU)
            #print('Pred position {:d}'.format(index))
            #print(pred_P_ENU)
            #print('...')

            # track current frame
            input_image = self.get_input_image(image_path.format(index), self.feed_width, self.feed_height)
            feature.track(np.asarray(input_image))

            # do optimization
            self.frame_ids.append(index)
            self.long_track_features = [i for i in feature.tracks if len(i) is len(feature.tracks[0])]
            self.optimize()

            # sliding window
            if index > self.WINDOW_SIZE:
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
        # compute jacobians and residuals
        self.compute_jacobian()
        # perform iterative optimization
        self.iterative_optimize()
        # set optimized states
        self.set_state()

    def slide_window(self):
        # optimize after 10 frames
        self.frame_ids.pop(0)
        self.slide_window()

    def get_state(self):
        self.states = np.array([])
        dim = len(self.frame_ids) * (8 + len(self.long_track_features))
        self.states.resize((dim, 1))
        self.frameid2state = {}

        index = 0
        frame_count = 0
        for frame_id in self.frame_ids:
            # position 3x1
            self.states[index:(index + 3)] = self.pred_poses[frame_id][:, [3]]
            self.frameid2state[frame_id] = index
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
        index = 0
        frame_count = 0
        for frame_id in self.frame_ids:
            # position 3x1
            self.pred_poses[frame_id][:, [3]] = self.states[index:(index + 3)]
            index += 3
            # orientation as rotation matrix 3x3
            q = quaternion.as_quat_array(self.states[index:(index + 4), 0])
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

    def compute_jacobian(self):
        """Compute jacobian matrix and residuals for optimization."""
        self.measure_count = len(self.long_track_features) * (len(self.frame_ids) - 1)
        self.jacobians = np.zeros((3 * self.measure_count, len(self.states)))
        self.residuals = np.zeros((3 * self.measure_count, 1))

        measure_index = 0

        for feature_count in range(len(self.long_track_features)):
            # start frame i
            frame_i_count = 0
            frame_j_count = frame_i_count
            frame_i_id = self.frame_ids[frame_i_count]
            index_i = self.frameid2state[frame_i_id]

            for frame_j_id in self.frame_ids[1:]:
                frame_j_count += 1
                ####################
                # parameters
                # frame i
                Pi = self.states[index_i:(index_i + 3)]
                Qi = quaternion.as_quat_array(self.states[(index_i + 3):(index_i + 7), 0])
                Ri = quaternion.as_rotation_matrix(Qi)
                scale_i = self.states[index_i + 7]
                depth_i = self.states[index_i + 8 + feature_count]
                # frame j
                index_j = self.frameid2state[frame_j_id]
                Pj = self.states[index_j:(index_j + 3)]
                Qj = quaternion.as_quat_array(self.states[(index_j + 3):(index_j + 7), 0])
                Rj = quaternion.as_rotation_matrix(Qj)
                scale_j = self.states[index_j + 7]
                depth_j = self.states[index_j + 8 + feature_count]

                ####################
                # residual
                # feature 3D coordinates estimated in frame i
                # P = K * p
                pts_homo_i = self.homo_coords(self.long_track_features[feature_count][frame_i_count])
                # P^cam = S * D * P
                pts_camera_i = pts_homo_i * depth_i  # * scale_i
                # X^imu = R^imu_cam * P^cam + T^imu_cam
                pts_imu_i = self.R_i_c.dot(pts_camera_i) + self.t_i_c
                # X^world = R^world_imu * X^imu + T^world_imu
                pts_w_i = Ri.dot(pts_imu_i) + Pi
                # feature 3D coordinates estimated in frame j
                pts_homo_j = self.homo_coords(self.long_track_features[feature_count][frame_j_count])
                pts_camera_j = pts_homo_j * depth_j  # * scale_j
                pts_imu_j = self.R_i_c.dot(pts_camera_j) + self.t_i_c
                pts_w_j = Rj.dot(pts_imu_j) + Pj
                # residual error
                self.residuals[measure_index:(measure_index + 3)] = pts_w_i - pts_w_j

                ####################
                # jacobians
                # jacobian of pose i (Pi and Qi)
                jaco_pose_i = np.zeros((3, 7))
                jaco_pose_i[:, :3] = np.identity(3)
                jaco_pose_i[:, 3:6] = -Ri.dot(vec2skew(pts_imu_i))
                self.jacobians[measure_index:(measure_index + 3), index_i:(index_i + 7)] = jaco_pose_i
                # jacobian of pose j (Pj and Qj)
                jaco_pose_j = np.zeros((3, 7))
                jaco_pose_j[:, :3] = -np.identity(3)
                jaco_pose_j[:, 3:6] = Rj.dot(vec2skew(pts_imu_j))
                self.jacobians[measure_index:(measure_index + 3), index_j:(index_j + 7)] = jaco_pose_j
                # jacobian of scale i
                # jaco_scale_i = Ri.dot(self.R_i_c).dot(pts_homo_i) * depth_i
                # self.jacobians[measure_index:(measure_index + 3), [index_i + 7]] = jaco_scale_i
                # jacobian of scale j
                # jaco_scale_j = -Rj.dot(self.R_i_c).dot(pts_homo_j) * depth_j
                # self.jacobians[measure_index:(measure_index + 3), [index_j + 7]] = jaco_scale_j
                # jacobian of depth i
                jaco_depth_i = Ri.dot(self.R_i_c).dot(pts_homo_i)  # * scale_i
                self.jacobians[measure_index:(measure_index + 3), [index_i + 8 + feature_count]] = jaco_depth_i
                # jacobian of depth j
                jaco_depth_j = -Rj.dot(self.R_i_c).dot(pts_homo_j)  # * scale_j
                self.jacobians[measure_index:(measure_index + 3), [index_j + 8 + feature_count]] = jaco_depth_j

                measure_index += 3

    def iterative_optimize(self):
        """Gauss-Newton iterative optimization."""
        num_iter = 0
        while np.dot(self.residuals.T, self.residuals)[0][0] > 0.3 and num_iter < 1:
            print('Iter {:d} before'.format(num_iter))
            print(np.dot(self.residuals.T, self.residuals)[0][0])

            H = np.dot(self.jacobians.T, self.jacobians) + 0.001 * np.identity(len(self.states))
            b = -np.dot(self.jacobians.T, self.residuals)

            delta = np.linalg.solve(H, b)
            # print(delta)
            # exit()
            # update states
            self.states = delta + self.states
            # update jacobians
            self.compute_jacobian()
            # 更新残差向量

            # print(len(self._state))
            # exit()
            print('Iter {:d} after'.format(num_iter))
            print(np.dot(self.residuals.T, self.residuals)[0][0])
            print('Params:')
            for frame_id in self.frame_ids:
                index = self.frameid2state[frame_id]
                s = self.states[index + 7]
                print(s)

            num_iter += 1

    def homo_coords(self, pt):
        """Convert 2D coordinate to 3D normalized, homogenous coordinates."""
        px = pt[0]
        py = pt[1]
        pz = 1.0
        p_homo = np.array([[px], [py], [pz]])
        p_homo = self.inv_K.dot(p_homo)

        return p_homo


if __name__ == '__main__':
    feature = tracker()
    post_slam = PostSLAM()
    post_slam.perform_slam(feature)
