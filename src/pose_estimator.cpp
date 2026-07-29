#include "pose_estimator.h"
#include <random>
#include <cmath>
#include <iostream>
#include <Eigen/Eigenvalues>

namespace energy_unit {

PoseEstimator::PoseEstimator(const EstimatorConfig& config)
    : config_(config) {
    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
        config_.fx, 0, config_.cx,
        0, config_.fy, config_.cy,
        0, 0, 1);
    dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
}

void PoseEstimator::updateIntrinsics(double fx, double fy, double cx, double cy) {
    config_.fx = fx; config_.fy = fy;
    config_.cx = cx; config_.cy = cy;
    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

bool PoseEstimator::loadModel(const std::string& obj_path, const std::string& m3t_dir) {
    (void)obj_path;
    std::string body_yaml = m3t_dir + "/Str3D_body.yaml";

    renderer_geometry_ = std::make_shared<m3t::RendererGeometry>("rg");
    color_camera_ = std::make_shared<SimpleColorCamera>("color_cam");
    depth_camera_ = std::make_shared<SimpleDepthCamera>("depth_cam");

    m3t::Intrinsics intr;
    intr.fu = static_cast<float>(config_.fx);
    intr.fv = static_cast<float>(config_.fy);
    intr.ppu = static_cast<float>(config_.cx);
    intr.ppv = static_cast<float>(config_.cy);
    intr.width = 640;
    intr.height = 480;
    color_camera_->set_intrinsics(intr);
    depth_camera_->set_intrinsics(intr);
    depth_camera_->set_depth_scale(0.001f);

    tracker_ = std::make_shared<m3t::Tracker>("tracker");

    // 🔥 预创建 MAX_TARGETS (3个) 独立的 M3T 实体槽位
    slots_.clear();
    for (int i = 0; i < MAX_TARGETS; ++i) {
        ObjectSlot slot;
        slot.id = i;
        std::string name_suffix = "_" + std::to_string(i);

        slot.body = std::make_shared<m3t::Body>("energy_unit" + name_suffix, std::filesystem::path(body_yaml));
        renderer_geometry_->AddBody(slot.body); // 所有物体加入同一个渲染器以计算相互遮挡

        auto region_model = std::make_shared<m3t::RegionModel>(
            "region_model" + name_suffix, slot.body, std::filesystem::path(m3t_dir) / "Str3D_region.bin");
        auto depth_model = std::make_shared<m3t::DepthModel>(
            "depth_model" + name_suffix, slot.body, std::filesystem::path(m3t_dir) / "Str3D_depth.bin");

        auto region_modality = std::make_shared<m3t::RegionModality>(
            "region_modality" + name_suffix, slot.body, color_camera_, region_model);
        auto depth_modality = std::make_shared<m3t::DepthModality>(
            "depth_modality" + name_suffix, slot.body, depth_camera_, depth_model);

        region_modality->MeasureOcclusions(depth_camera_);
        depth_modality->MeasureOcclusions();

        slot.link = std::make_shared<m3t::Link>("link" + name_suffix, slot.body);
        slot.link->AddModality(region_modality);
        slot.link->AddModality(depth_modality);

        slot.optimizer = std::make_shared<m3t::Optimizer>("optimizer" + name_suffix, slot.link);
        tracker_->AddOptimizer(slot.optimizer);

        m3t::Transform3fA init_pose = m3t::Transform3fA::Identity();
        init_pose.translation() = Eigen::Vector3f(0.0f, 0.0f, 10.0f); // 初始置于远处
        slot.body->set_body2world_pose(init_pose);
        slot.link->set_link2world_pose(init_pose);

        slot.active = false;
        slots_.push_back(slot);
    }

    if (!tracker_->SetUp()) {
        std::cerr << "[PoseEstimator] 多目标 M3T SetUp 失败" << std::endl;
        return false;
    }

    m3t_initialized_ = true;
    std::cout << "[PoseEstimator] 多目标 M3T (3槽位) 初始化成功！" << std::endl;
    return true;
}

std::vector<PoseResult> PoseEstimator::estimateMulti(
    const std::vector<Detection>& detections,
    const cv::Mat& depth,
    const cv::Mat& color_bgr) {

    std::vector<PoseResult> results;
    if (!m3t_initialized_) return results;

    // 1. 灌入最新图像
    color_camera_->set_image(color_bgr);
    depth_camera_->set_image(depth);

    // 2. 预解析当前 YOLO 检测框的 PCA 位姿
    struct DetInfo {
        Detection det;
        bool pca_valid = false;
        m3t::Transform3fA pca_pose = m3t::Transform3fA::Identity();
        Eigen::Vector3d center_3d = Eigen::Vector3d::Zero();
        bool matched = false;
    };

    std::vector<DetInfo> det_infos;
    for (const auto& det : detections) {
        DetInfo info;
        info.det = det;
        info.pca_pose = computePCAPose(det, depth, info.pca_valid, info.center_3d);
        det_infos.push_back(info);
    }

    // 3. 数据关联：匹配已激活槽位与当前 YOLO 检测
    for (auto& slot : slots_) {
        if (!slot.active) continue;

        double min_dist = 0.3; // 3D 距离关联阈值 (0.3米)
        int best_match_idx = -1;

        for (size_t i = 0; i < det_infos.size(); ++i) {
            if (det_infos[i].matched || !det_infos[i].pca_valid) continue;
            double d = (slot.last_center - det_infos[i].center_3d).norm();
            if (d < min_dist) {
                min_dist = d;
                best_match_idx = (int)i;
            }
        }

        if (best_match_idx != -1) {
            det_infos[best_match_idx].matched = true;
            slot.loss_counter = 0;
        } else {
            slot.loss_counter++;
            if (slot.loss_counter >= 5) { // 连续 5 帧未匹配到则注销槽位
                resetTrackingSlot(slot.id);
            }
        }
    }

    // 4. 新目标激活：将未匹配的 YOLO 检测分配给空闲槽位
    for (auto& info : det_infos) {
        if (info.matched || !info.pca_valid) continue;

        for (auto& slot : slots_) {
            if (!slot.active) {
                slot.body->set_body2world_pose(info.pca_pose);
                slot.link->set_link2world_pose(info.pca_pose);
                slot.last_center = info.center_3d;
                slot.active = true;
                slot.loss_counter = 0;
                info.matched = true;
                break;
            }
        }
    }

    // 5. 统一执行 M3T Step 求解
    bool any_active = false;
    for (const auto& slot : slots_) {
        if (slot.active) { any_active = true; break; }
    }

    if (any_active) {
        tracker_->ExecuteTrackingStep(0); // 内部并行/联合优化所有活跃槽位
    }

    // 6. 提取所有激活槽位的 6D 位姿结果
    for (auto& slot : slots_) {
        if (!slot.active) continue;

        m3t::Transform3fA pose = slot.body->body2world_pose();
        Eigen::Matrix3d R = pose.linear().cast<double>();
        Eigen::Vector3d t = pose.translation().cast<double>();

        // 防漂移异常值二次校验
        if (t.z() < 0.1 || t.z() > 6.0 || std::isnan(t.x())) {
            resetTrackingSlot(slot.id);
            continue;
        }

        slot.last_center = t;

        PoseResult res;
        res.T_cam_obj = Eigen::Matrix4d::Identity();
        res.T_cam_obj.block<3, 3>(0, 0) = R;
        res.T_cam_obj.block<3, 1>(0, 3) = t;
        res.translation = t;
        res.quaternion = Eigen::Quaterniond(R);
        res.quaternion.normalize();
        res.axis_direction = R.col(2);
        res.euler_angles = R.eulerAngles(2, 1, 0) * 180.0 / M_PI;
        res.confidence = 1.0;
        res.valid = true;

        results.push_back(res);
    }

    return results;
}

m3t::Transform3fA PoseEstimator::computePCAPose(
    const Detection& det, const cv::Mat& depth, bool& valid, Eigen::Vector3d& out_center) const {

    valid = false;
    m3t::Transform3fA pose = m3t::Transform3fA::Identity();

    auto points_3d = extractPointCloud(det, depth);
    if (points_3d.size() < 10) return pose;

    Eigen::Vector3d centroid(0, 0, 0);
    for (const auto& p : points_3d) centroid += Eigen::Vector3d(p.x, p.y, p.z);
    centroid /= points_3d.size();
    out_center = centroid;

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : points_3d) {
        Eigen::Vector3d diff = Eigen::Vector3d(p.x, p.y, p.z) - centroid;
        cov += diff * diff.transpose();
    }
    cov /= points_3d.size();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) return pose;

    Eigen::Vector3d z_axis = solver.eigenvectors().col(2).normalized();
    if (z_axis.y() < 0) z_axis = -z_axis;

    Eigen::Vector3d cam_x(1.0, 0.0, 0.0);
    Eigen::Vector3d x_proj = cam_x - cam_x.dot(z_axis) * z_axis;
    if (x_proj.norm() < 1e-4) {
        x_proj = Eigen::Vector3d(0.0, 1.0, 0.0).cross(z_axis);
    }
    Eigen::Vector3d x_axis = x_proj.normalized();
    Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    x_axis = y_axis.cross(z_axis).normalized();

    pose.linear() << x_axis.cast<float>(), y_axis.cast<float>(), z_axis.cast<float>();
    pose.translation() = centroid.cast<float>();

    if (centroid.z() > 0.2 && centroid.z() < 5.0) {
        valid = true;
    }
    return pose;
}

std::vector<cv::Point3f> PoseEstimator::extractPointCloud(const Detection& det, const cv::Mat& depth) const {
    std::vector<cv::Point3f> points;
    cv::Rect roi = det.bbox_axis_aligned & cv::Rect(0, 0, depth.cols, depth.rows);
    if (roi.width <= 0 || roi.height <= 0) return points;

    int area = roi.width * roi.height;
    int step = (area > 40000) ? 3 : (area > 10000 ? 2 : 1);
    points.reserve(area / (step * step));

    const bool has_mask = !det.mask.empty();
    for (int y = roi.y; y < roi.y + roi.height; y += step) {
        for (int x = roi.x; x < roi.x + roi.width; x += step) {
            if (has_mask && det.mask.at<uchar>(y - roi.y, x - roi.x) == 0) continue;

            uint16_t d = depth.at<uint16_t>(y, x);
            if (d > 100 && d < 5000) {
                double z_m = d / 1000.0;
                double x_m = (x - config_.cx) * z_m / config_.fx;
                double y_m = (y - config_.cy) * z_m / config_.fy;
                points.emplace_back(static_cast<float>(x_m), static_cast<float>(y_m), static_cast<float>(z_m));
            }
        }
    }
    return points;
}

void PoseEstimator::drawPoseAxes(cv::Mat& image, const PoseResult& pose, double axis_length) const {
    if (!pose.valid) return;

    std::vector<cv::Point3f> axis_pts_3d = {
        cv::Point3f(0, 0, 0), cv::Point3f(axis_length, 0, 0),
        cv::Point3f(0, axis_length, 0), cv::Point3f(0, 0, axis_length)
    };

    cv::Mat rvec, tvec;
    Eigen::Matrix3d R = pose.T_cam_obj.block<3, 3>(0, 0);
    cv::Mat R_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R_cv.at<double>(i, j) = R(i, j);
    cv::Rodrigues(R_cv, rvec);

    tvec = (cv::Mat_<double>(3, 1) << pose.translation.x(), pose.translation.y(), pose.translation.z());

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(axis_pts_3d, rvec, tvec, camera_matrix_, dist_coeffs_, img_pts);

    cv::circle(image, img_pts[0], 5, cv::Scalar(255, 255, 255), -1);
    cv::arrowedLine(image, img_pts[0], img_pts[1], cv::Scalar(0, 0, 255), 2);
    cv::arrowedLine(image, img_pts[0], img_pts[2], cv::Scalar(0, 255, 0), 2);
    cv::arrowedLine(image, img_pts[0], img_pts[3], cv::Scalar(255, 0, 0), 2);
}

void PoseEstimator::resetTrackingSlot(int slot_id) {
    if (slot_id < 0 || slot_id >= (int)slots_.size()) return;

    auto& slot = slots_[slot_id];
    slot.active = false;
    slot.loss_counter = 0;

    m3t::Transform3fA clean_pose = m3t::Transform3fA::Identity();
    clean_pose.translation() = Eigen::Vector3f(0.0f, 0.0f, 10.0f);
    slot.body->set_body2world_pose(clean_pose);
    slot.link->set_link2world_pose(clean_pose);
}

void PoseEstimator::resetAllTracking() {
    for (int i = 0; i < (int)slots_.size(); ++i) {
        resetTrackingSlot(i);
    }
    if (tracker_) tracker_->SetUp();
}

}  // namespace energy_unit