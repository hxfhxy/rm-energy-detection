#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <cmath>

namespace energy_unit {

struct Detection;   // 假设已在 detector 中定义

struct EstimatorConfig {
    // 相机内参
    double fx = 424.49407958984375;
    double fy = 424.49407958984375;
    double cx = 422.3330383300781;
    double cy = 241.14089965820312;

    // 圆柱半径（用于生成模型点云）
    double cylinder_radius = 0.0475;   // m
    double cylinder_height = 0.150;    // m

    // 滤波参数
    double voxel_leaf_size = 0.01;    // 体素下采样尺寸（m）
    double cluster_tolerance = 0.015;  // 欧式聚类距离阈值（m）
    int cluster_min_size = 50;         // 最小簇点数
    int stat_outlier_k = 20;           // 统计滤波近邻数
    double stat_std_mul = 1.0;         // 标准差倍数

    // ICP 参数
    int icp_max_iter = 5;
    double icp_max_correspondence_dist = 0.05;  // 对应点最大距离（m）
    double trimmed_ratio = 0.75;                // Trimmed ICP 保留点比例
    int num_roll_candidates = 4;               // 绕轴 roll 候选数

    // 调试
    bool verbose = false;
};

struct PoseResult {
    bool valid = false;
    int label = 0;
    double confidence = 1.0;
    Eigen::Vector3d translation;
    Eigen::Quaterniond quaternion;
    Eigen::Vector3d axis_direction;   // Z轴方向
    Eigen::Vector3d euler_angles;     // 度
    Eigen::Matrix4d T_cam_obj;        // 相机坐标系下的物体位姿
};

class PoseEstimator {
public:
    PoseEstimator(const EstimatorConfig& config);
    void updateIntrinsics(double fx, double fy, double cx, double cy);

    // 主接口
    std::vector<PoseResult> estimateMulti(const std::vector<Detection>& detections,
                                          const cv::Mat& depth,
                                          const cv::Mat& color_bgr);

    // 可视化
    void drawPoseAxes(cv::Mat& image, const PoseResult& pose, double axis_length = 0.08) const;

private:
    EstimatorConfig config_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    // 模型点云（物体坐标系）
    std::vector<Eigen::Vector3d> model_cloud_;
    void buildModelCloud();          // 根据尺寸生成圆柱点云

    // 点云处理
    std::vector<Eigen::Vector3d> extractPointCloud(const Detection& det, const cv::Mat& depth) const;
    std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3d>& pts, double leaf) const;
    std::vector<Eigen::Vector3d> statisticalOutlierRemoval(const std::vector<Eigen::Vector3d>& pts, int k, double std_mul) const;
    std::vector<Eigen::Vector3d> euclideanClusterSelect(const std::vector<Eigen::Vector3d>& pts, double tolerance, int min_size) const;

    // PCA
    Eigen::Vector3d computePCALongAxis(const std::vector<Eigen::Vector3d>& pts, Eigen::Vector3d& centroid) const;

    // 候选生成
    std::vector<Eigen::Matrix4d> generateInitialCandidates(const Eigen::Vector3d& centroid,
                                                           const Eigen::Vector3d& long_axis,
                                                           int num_rolls) const;

    // Trimmed ICP
    struct ICPResult {
        Eigen::Matrix4d T;
        double trimmed_rmse;
        double correspondence_ratio;
    };
    ICPResult trimmedICP(const std::vector<Eigen::Vector3d>& source,
                         const std::vector<Eigen::Vector3d>& target,
                         const Eigen::Matrix4d& init_T,
                         double max_dist,
                         int max_iter,
                         double trim_ratio) const;

    // 追踪相关（保留EMA平滑）
    struct Target {
        bool active = false;
        int loss_counter = 0;
        int label = 0;
        Eigen::Vector3d center;
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    };
    std::deque<Target> targets_;    // 简单固定大小窗口
    static constexpr int MAX_TARGETS = 5;
};

} // namespace energy_unit