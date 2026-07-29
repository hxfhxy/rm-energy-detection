#ifndef POSE_ESTIMATOR_H_
#define POSE_ESTIMATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/depth_modality.h>
#include <m3t/depth_model.h>
#include <m3t/link.h>
#include <m3t/optimizer.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/renderer_geometry.h>
#include <m3t/tracker.h>

#include "energy_unit_detector.h"

namespace energy_unit {

struct EstimatorConfig {
    double fx = 615.0;
    double fy = 615.0;
    double cx = 320.0;
    double cy = 240.0;
    double cylinder_radius = 0.04;
    double cylinder_height = 0.12;
    int ransac_iterations = 100;
    double ransac_threshold = 0.01;
};

struct PoseResult {
    bool valid = false;
    Eigen::Matrix4d T_cam_obj;
    Eigen::Vector3d translation;
    Eigen::Quaterniond quaternion;
    Eigen::Vector3d axis_direction;
    Eigen::Vector3d euler_angles;
    double confidence = 0.0;
};

// 🚀 M3T 追踪槽位封装结构体
struct ObjectSlot {
    int id = 0;
    std::shared_ptr<m3t::Body> body;
    std::shared_ptr<m3t::Link> link;
    std::shared_ptr<m3t::Optimizer> optimizer;

    bool active = false;          // 当前槽位是否处于活跃追踪状态
    int loss_counter = 0;         // 连续未匹配帧数计数
    Eigen::Vector3d last_center;  // 上一帧物体的 3D 质心
};

// 🚀 自定义虚拟彩色相机
class SimpleColorCamera : public m3t::ColorCamera {
public:
    SimpleColorCamera(const std::string& name) : m3t::ColorCamera(name) {}

    bool SetUp() override {
        set_up_ = true;
        return true;
    }

    bool UpdateImage(bool /*synchronized*/) override { return true; }

    void set_image(const cv::Mat& image) {
        image_ = image;
        save_index_ = 0;
    }

    void set_intrinsics(const m3t::Intrinsics& intrinsics) {
        intrinsics_ = intrinsics;
    }
};

// 🚀 自定义虚拟深度相机
class SimpleDepthCamera : public m3t::DepthCamera {
public:
    SimpleDepthCamera(const std::string& name) : m3t::DepthCamera(name) {}

    bool SetUp() override {
        set_up_ = true;
        return true;
    }

    bool UpdateImage(bool /*synchronized*/) override { return true; }

    void set_image(const cv::Mat& image) {
        image_ = image;
        save_index_ = 0;
    }

    void set_intrinsics(const m3t::Intrinsics& intrinsics) {
        intrinsics_ = intrinsics;
    }

    void set_depth_scale(float depth_scale) {
        depth_scale_ = depth_scale;
    }
};

class PoseEstimator {
public:
    explicit PoseEstimator(const EstimatorConfig& config);
    void updateIntrinsics(double fx, double fy, double cx, double cy);
    bool loadModel(const std::string& obj_path, const std::string& m3t_dir);

    // 🔥 多目标位姿求解主接口
    std::vector<PoseResult> estimateMulti(
        const std::vector<Detection>& detections,
        const cv::Mat& depth,
        const cv::Mat& color_bgr);

    void drawPoseAxes(cv::Mat& image, const PoseResult& pose, double axis_length = 0.1) const;
    void resetTrackingSlot(int slot_id);
    void resetAllTracking();

private:
    static constexpr int MAX_TARGETS = 3; // 预分配最多 3 个独立槽位
    std::vector<ObjectSlot> slots_;

    // 辅助计算函数
    m3t::Transform3fA computePCAPose(
        const Detection& det, const cv::Mat& depth, bool& valid, Eigen::Vector3d& out_center) const;
    std::vector<cv::Point3f> extractPointCloud(const Detection& det, const cv::Mat& depth) const;

    EstimatorConfig config_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    std::shared_ptr<m3t::RendererGeometry> renderer_geometry_;
    std::shared_ptr<SimpleColorCamera> color_camera_;
    std::shared_ptr<SimpleDepthCamera> depth_camera_;
    std::shared_ptr<m3t::Tracker> tracker_;

    bool m3t_initialized_ = false;
};

}  // namespace energy_unit

#endif  // POSE_ESTIMATOR_H_