/**
 * @file energy_unit_node.cpp
 * @brief 能量单元极速解算节点 (无 TF、无 RViz、纯终端与 OpenCV 输出)
 */

#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>

#include "energy_unit_detector.h"
#include "pose_estimator.h"

using namespace std::chrono_literals;

namespace energy_unit {

struct StampedColorFrame {
    std::int64_t stamp_ns;
    cv::Mat image;
};

struct StampedDepthFrame {
    std::int64_t stamp_ns;
    cv::Mat image;
};

static std::int64_t StampToNanoseconds(const builtin_interfaces::msg::Time &stamp) {
    return static_cast<std::int64_t>(stamp.sec) * 1000000000LL + static_cast<std::int64_t>(stamp.nanosec);
}

template <typename Frame>
static std::optional<Frame> FindNearestFrame(
    const std::deque<Frame> &frames, std::int64_t target_ns, std::int64_t tolerance_ns) {
    std::optional<Frame> best_frame;
    std::int64_t best_delta = std::numeric_limits<std::int64_t>::max();
    for (const auto &frame : frames) {
        const auto delta = frame.stamp_ns > target_ns ? frame.stamp_ns - target_ns : target_ns - frame.stamp_ns;
        if (delta < best_delta) {
            best_delta = delta;
            best_frame = frame;
        }
    }
    if (!best_frame || best_delta > tolerance_ns) return std::nullopt;
    return best_frame;
}

template <typename Frame>
static void PruneHistory(std::deque<Frame> *frames, std::size_t max_size) {
    while (frames->size() > max_size) frames->pop_front();
}

class EnergyUnitNode : public rclcpp::Node {
public:
    EnergyUnitNode() : Node("energy_unit_node") {
        this->declare_parameter("sync_tolerance_ms", 10.0); // 严格同步，防止掉帧
        sync_tolerance_ns_ = static_cast<std::int64_t>(this->get_parameter("sync_tolerance_ms").as_double() * 1000000.0);

        // 1. 初始化 YOLO-Seg 检测器 (把你的模型绝对路径直接写死)
        DetectorConfig det_cfg;
        det_cfg.model_path = "/home/hzy/能量单元位姿识别/model/yolo/best.onnx"; // ⚠️ 请确保这是你真实的绝对路径
        detector_ = std::make_unique<EnergyUnitDetector>(det_cfg);

        // 2. 直接硬编码【深度相机】内参 (用于从深度图生成 3D 点云)
        EstimatorConfig est_cfg;
        est_cfg.fx = 424.49407958984375; 
        est_cfg.fy = 424.49407958984375;
        est_cfg.cx = 422.3330383300781;
        est_cfg.cy = 241.14089965820312;
        est_cfg.cylinder_radius = 0.0475; // 95mm 的一半
        est_cfg.cylinder_height = 0.150;  // 整体高度 150mm 
        
        RCLCPP_INFO(this->get_logger(), "已硬编码深度相机内参: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", 
                    est_cfg.fx, est_cfg.fy, est_cfg.cx, est_cfg.cy);

        estimator_ = std::make_unique<PoseEstimator>(est_cfg);

        // 3. 订阅 rosbag 话题
        color_compressed_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/camera/camera/color/image_raw/compressed", rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) { SetColorImageCompressed(msg); });
            
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/depth/image_rect_raw", rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) { SetDepthImage(msg); });

        RCLCPP_INFO(this->get_logger(), "== 等待 rosbag play ==");
    }

private:
    std::unique_ptr<EnergyUnitDetector> detector_;
    std::unique_ptr<PoseEstimator> estimator_;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_compressed_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

    mutable std::mutex mutex_;
    std::deque<StampedColorFrame> color_history_;
    std::deque<StampedDepthFrame> depth_history_;
    std::int64_t sync_tolerance_ns_;
    std::int64_t last_processed_stamp_ns_ = 0;

    void SetColorImageCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr &msg) {
        cv::Mat image;
        try {
            image = cv::imdecode(cv::Mat(1, msg->data.size(), CV_8UC1, (void*)msg->data.data()), cv::IMREAD_COLOR);
            if (image.empty()) return;
        } catch (...) { return; }
        
        auto stamp_ns = StampToNanoseconds(msg->header.stamp);
        {
            const std::lock_guard<std::mutex> lock{mutex_};
            color_history_.push_back({stamp_ns, image}); 
            PruneHistory(&color_history_, 30);
        }
        TryProcessFrame(stamp_ns);
    }

    void SetDepthImage(const sensor_msgs::msg::Image::SharedPtr &msg) {
        cv::Mat image;
        try {
            // 使用 toCvShare 浅拷贝代替 toCvCopy 深拷贝，提升性能
            if (msg->encoding == "16UC1") {
                image = cv_bridge::toCvShare(msg, "16UC1")->image;
            } else if (msg->encoding == "32FC1") {
                cv::Mat f = cv_bridge::toCvShare(msg, "32FC1")->image;
                f.convertTo(image, CV_16UC1, 1000.0);
            } else {
                image = cv_bridge::toCvShare(msg, msg->encoding)->image;
            }
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "深度图转换失败: %s", e.what());
            return;
        }
        if (image.empty()) return;
        
        auto stamp_ns = StampToNanoseconds(msg->header.stamp);
        {
            const std::lock_guard<std::mutex> lock{mutex_};
            depth_history_.push_back({stamp_ns, image.clone()}); 
            PruneHistory(&depth_history_, 30);
        }
        TryProcessFrame(stamp_ns);
    }

    void TryProcessFrame(std::int64_t trigger_stamp_ns) {
        std::optional<StampedColorFrame> color;
        std::optional<StampedDepthFrame> depth;
        {
            const std::lock_guard<std::mutex> lock{mutex_};
            if (color_history_.empty() || depth_history_.empty()) return;
            color = FindNearestFrame(color_history_, trigger_stamp_ns, sync_tolerance_ns_);
            depth = FindNearestFrame(depth_history_, trigger_stamp_ns, sync_tolerance_ns_);
        }
        if (!color || !depth) return;

        // 彻底解决双重触发导致卡顿的问题
        if (color->stamp_ns == last_processed_stamp_ns_) return; 
        last_processed_stamp_ns_ = color->stamp_ns;

        processFrame(color->image, depth->image);
    }
    
    cv::Mat reprojectDepthToRGB(const cv::Mat& depth_16u,        // 深度图 (mm)
                                const cv::Mat& color_8u3,        // 原始彩色图
                                const Eigen::Matrix3d& K_depth,
                                const Eigen::Matrix3d& K_rgb,
                                const Eigen::Matrix3d& R_d2c = Eigen::Matrix3d::Identity(),
                                const Eigen::Vector3d& t_d2c = Eigen::Vector3d::Zero()) {
        cv::Mat uv_rgb(depth_16u.rows, depth_16u.cols, CV_8UC3, cv::Scalar(0,0,0));

        // 预计算深度内参逆矩阵
        double inv_fx_d = 1.0 / K_depth(0,0);
        double inv_fy_d = 1.0 / K_depth(1,1);
        double cx_d = K_depth(0,2);
        double cy_d = K_depth(1,2);

        // 彩色内参
        double fx_rgb = K_rgb(0,0);
        double fy_rgb = K_rgb(1,1);
        double cx_rgb = K_rgb(0,2);
        double cy_rgb = K_rgb(1,2);

        for (int vd = 0; vd < depth_16u.rows; ++vd) {
            const uint16_t* drow = depth_16u.ptr<uint16_t>(vd);
            cv::Vec3b* urrow = uv_rgb.ptr<cv::Vec3b>(vd);
            for (int ud = 0; ud < depth_16u.cols; ++ud) {
                uint16_t d_mm = drow[ud];
                if (d_mm < 100 || d_mm > 5000) continue;   // 无效深度

                // 1. 反投影到深度相机坐标系
                double z = d_mm / 1000.0;
                double x = (ud - cx_d) * z * inv_fx_d;
                double y = (vd - cy_d) * z * inv_fy_d;
                Eigen::Vector3d P_d(x, y, z);

                // 2. 转换到彩色相机坐标系 (假设 R=I, t=0)
                Eigen::Vector3d P_c = R_d2c * P_d + t_d2c;

                // 3. 投影到彩色图像平面
                double uc = (P_c.x() * fx_rgb / P_c.z()) + cx_rgb;
                double vc = (P_c.y() * fy_rgb / P_c.z()) + cy_rgb;
                int iuc = std::lround(uc);
                int ivc = std::lround(vc);

                // 4. 采样彩色图
                if (iuc >= 0 && iuc < color_8u3.cols && ivc >= 0 && ivc < color_8u3.rows) {
                    urrow[ud] = color_8u3.at<cv::Vec3b>(ivc, iuc);
                }
            }
        }
        return uv_rgb;
    }

    void processFrame(const cv::Mat& color, const cv::Mat& depth) {
        // ---------- 生成 uv-rgb 图像 (与深度图对齐) ----------
        static const Eigen::Matrix3d K_depth(
            (Eigen::Matrix3d() << 424.49407958984375, 0.0, 422.3330383300781,
                                0.0, 424.49407958984375, 241.14089965820312,
                                0.0, 0.0, 1.0).finished());
        static const Eigen::Matrix3d K_rgb(
            (Eigen::Matrix3d() << 920.73095703125, 0.0, 646.6663818359375,
                                0.0, 921.03125, 346.5528564453125,
                                0.0, 0.0, 1.0).finished());

        cv::Mat uv_rgb = reprojectDepthToRGB(depth, color, K_depth, K_rgb);
        // 如果以后获取了外参，可改为：
        // cv::Mat uv_rgb = reprojectDepthToRGB(depth, color, K_depth, K_rgb, R_d2c, t_d2c);

        // ---------- 检测与姿态估计 ----------
        auto t_start = std::chrono::high_resolution_clock::now();

        // 用 uv_rgb 进行检测（尺寸与 depth 一致，掩码直接适配深度图像素）
        auto detections = detector_->detect(uv_rgb, depth);
        auto valid_poses = estimator_->estimateMulti(detections, depth, uv_rgb);

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double current_fps = 1000.0 / total_ms;

        // ---------- 可视化 ----------
        cv::Mat display = uv_rgb.clone();
        detector_->drawDetections(display, detections);

        // OSD 文字
        char osd_text[128];
        snprintf(osd_text, sizeof(osd_text), "FPS: %5.1f | Time: %5.1f ms", current_fps, total_ms);
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(osd_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
        cv::Point textOrg(15, 30);
        cv::Rect textBg(textOrg.x - 5, textOrg.y - textSize.height - 5, textSize.width + 10, textSize.height + 10);
        cv::rectangle(display, textBg, cv::Scalar(0, 0, 0), -1);
        cv::putText(display, osd_text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        printf("\r[FPS: %5.1f] 耗时: %5.1f ms | 目标数: %zu ", current_fps, total_ms, valid_poses.size());
        fflush(stdout);

        for (size_t i = 0; i < valid_poses.size(); ++i) {
            estimator_->drawPoseAxes(display, valid_poses[i]);
            printf("\n  [Target %zu] 坐标(XYZ): %.3f, %.3f, %.3f | 姿态(RPY): %.1f, %.1f, %.1f",
                i,
                valid_poses[i].translation.x(), valid_poses[i].translation.y(), valid_poses[i].translation.z(),
                valid_poses[i].euler_angles.x(), valid_poses[i].euler_angles.y(), valid_poses[i].euler_angles.z());
        }

        // 深度图可视化
        cv::Mat depth_display;
        depth.convertTo(depth_display, CV_8UC1, 255.0 / 4000.0);
        cv::imshow("Depth Stream", depth_display);

        // 显示主要窗口 (uv_rgb + 检测 + 坐标轴)
        cv::imshow("Energy Unit RANSAC Tracker", display);

        // 对齐检查窗口：uv_rgb 与深度图叠加（尺寸相同）
        cv::Mat alignment_check;
        cv::cvtColor(depth_display, depth_display, cv::COLOR_GRAY2BGR); // 转为3通道以便叠加
        cv::addWeighted(display, 0.5, depth_display, 0.5, 0, alignment_check);
        cv::imshow("Alignment Check (UV-RGB + Depth)", alignment_check);

        cv::waitKey(1);
    }
};


}  // namespace energy_unit

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<energy_unit::EnergyUnitNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}