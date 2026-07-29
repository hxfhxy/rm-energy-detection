/**
 * @file energy_unit_node.cpp
 * @brief 能量单元位姿识别 ROS2 节点（支持多目标解析）
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "energy_unit_detector.h"
#include "pose_estimator.h"
#include "arm_transform.h"

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

static std::int64_t StampToNanoseconds(
    const builtin_interfaces::msg::Time &stamp) {
    return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
           static_cast<std::int64_t>(stamp.nanosec);
}

template <typename Frame>
static std::optional<Frame> FindNearestFrame(
    const std::deque<Frame> &frames,
    std::int64_t target_ns,
    std::int64_t tolerance_ns) {
    std::optional<Frame> best_frame;
    std::int64_t best_delta = std::numeric_limits<std::int64_t>::max();
    for (const auto &frame : frames) {
        const auto delta = frame.stamp_ns > target_ns
                               ? frame.stamp_ns - target_ns
                               : target_ns - frame.stamp_ns;
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
        this->declare_parameter("model_path", std::string("model/yolo/best.onnx"));
        this->declare_parameter("conf_threshold", 0.6);
        this->declare_parameter("nms_threshold", 0.45);
        this->declare_parameter("input_size", 640);
        this->declare_parameter("min_area", 2000.0);
        this->declare_parameter("max_area", 400000.0);
        this->declare_parameter("cylinder_radius", 0.04);
        this->declare_parameter("cylinder_height", 0.12);
        this->declare_parameter("hand_eye_roll", 0.0);
        this->declare_parameter("hand_eye_pitch", 0.0);
        this->declare_parameter("hand_eye_yaw", 0.0);
        this->declare_parameter("hand_eye_tx", 0.0);
        this->declare_parameter("hand_eye_ty", 0.0);
        this->declare_parameter("hand_eye_tz", 0.0);
        this->declare_parameter("result_dir", std::string("result"));
        this->declare_parameter("save_interval", 5);
        this->declare_parameter("frame_history_size", 100);
        this->declare_parameter("sync_tolerance_ms", 50.0);

        const auto history_size = std::max<long>(
            1, this->get_parameter("frame_history_size").as_int());
        max_history_size_ = static_cast<std::size_t>(history_size);
        const auto tolerance_ms =
            this->get_parameter("sync_tolerance_ms").as_double();
        sync_tolerance_ns_ = static_cast<std::int64_t>(tolerance_ms * 1000000.0);

        DetectorConfig det_cfg;
        det_cfg.model_path = this->get_parameter("model_path").as_string();
        det_cfg.conf_threshold = static_cast<float>(this->get_parameter("conf_threshold").as_double());
        det_cfg.nms_threshold = static_cast<float>(this->get_parameter("nms_threshold").as_double());
        det_cfg.input_size = static_cast<int>(this->get_parameter("input_size").as_int());
        det_cfg.min_area = this->get_parameter("min_area").as_double();
        det_cfg.max_area = this->get_parameter("max_area").as_double();
        detector_ = std::make_unique<EnergyUnitDetector>(det_cfg);

        EstimatorConfig est_cfg;
        est_cfg.cylinder_radius = this->get_parameter("cylinder_radius").as_double();
        est_cfg.cylinder_height = this->get_parameter("cylinder_height").as_double();
        estimator_ = std::make_unique<PoseEstimator>(est_cfg);

        std::string model_dir = det_cfg.model_path.substr(0, det_cfg.model_path.find_last_of('/'));
        std::string model_root = model_dir;
        size_t last_slash = model_dir.find_last_of('/');
        if (last_slash != std::string::npos) model_root = model_dir.substr(0, last_slash);

        m3t_dir_ = "/home/hzy/能量单元位姿识别/model/m3t";
        obj_path_ = model_root + "/obj/Str3D.obj";

        Transform T_ee_cam = ArmTransform::fromEuler(
            this->get_parameter("hand_eye_roll").as_double(),
            this->get_parameter("hand_eye_pitch").as_double(),
            this->get_parameter("hand_eye_yaw").as_double(),
            this->get_parameter("hand_eye_tx").as_double(),
            this->get_parameter("hand_eye_ty").as_double(),
            this->get_parameter("hand_eye_tz").as_double());
        arm_transform_ = std::make_unique<ArmTransform>(Transform(), T_ee_cam);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) { SetColorImage(msg); });
        color_compressed_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/camera/camera/color/image_raw/compressed", rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) { SetColorImageCompressed(msg); });
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/depth/image_rect_raw", rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) { SetDepthImage(msg); });
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info", rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) { SetCameraInfo(msg); });

        result_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/energy_unit/result_image", 10);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/energy_unit/pose", 10);
        arm_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/energy_unit/arm_pose", 10);

        result_dir_ = this->get_parameter("result_dir").as_string();
        save_interval_ = static_cast<int>(this->get_parameter("save_interval").as_int());
        std::filesystem::create_directories(result_dir_);
        std::filesystem::create_directories(result_dir_ + "/images");
        frame_count_ = 0;

        pose_csv_.open(result_dir_ + "/poses.csv");
        pose_csv_ << "frame,timestamp_ns,target_idx,tx,ty,tz,qw,qx,qy,qz,"
                  << "roll_deg,pitch_deg,yaw_deg,"
                  << "detect_ms,total_ms,num_objects\n";

        RCLCPP_INFO(this->get_logger(), "多目标能量单元位姿识别节点已启动");
    }

private:
    std::unique_ptr<EnergyUnitDetector> detector_;
    std::unique_ptr<PoseEstimator> estimator_;
    std::unique_ptr<ArmTransform> arm_transform_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string result_dir_;
    int save_interval_;
    int frame_count_;
    std::ofstream pose_csv_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_compressed_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_image_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr arm_pose_pub_;

    mutable std::mutex mutex_;
    std::deque<StampedColorFrame> color_history_;
    std::deque<StampedDepthFrame> depth_history_;
    std::size_t max_history_size_{100};
    std::int64_t sync_tolerance_ns_{50000000};
    bool camera_info_received_ = false;

    std::string obj_path_;
    std::string m3t_dir_;

    void SetColorImage(const sensor_msgs::msg::Image::SharedPtr &msg) {
        cv::Mat image;
        try {
            image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "彩色图转换失败: %s", e.what());
            return;
        }
        const auto stamp_ns = StampToNanoseconds(msg->header.stamp);
        {
            const std::lock_guard<std::mutex> lock{mutex_};
            color_history_.push_back({stamp_ns, image.clone()});
            PruneHistory(&color_history_, max_history_size_);
        }
        TryProcessFrame(stamp_ns);
    }

    void SetColorImageCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr &msg) {
        cv::Mat image;
        try {
            cv::Mat buf(1, static_cast<int>(msg->data.size()), CV_8UC1,
                        const_cast<uint8_t *>(msg->data.data()));
            image = cv::imdecode(buf, cv::IMREAD_COLOR);
            if (image.empty()) return;
        } catch (...) { return; }
        const auto stamp_ns = StampToNanoseconds(msg->header.stamp);
        {
            const std::lock_guard<std::mutex> lock{mutex_};
            color_history_.push_back({stamp_ns, image.clone()});
            PruneHistory(&color_history_, max_history_size_);
        }
        TryProcessFrame(stamp_ns);
    }

    void SetDepthImage(const sensor_msgs::msg::Image::SharedPtr &msg) {
        cv::Mat image;
        try {
            if (msg->encoding == "16UC1") {
                image = cv_bridge::toCvCopy(msg, "16UC1")->image;
            } else if (msg->encoding == "32FC1") {
                cv::Mat f = cv_bridge::toCvCopy(msg, "32FC1")->image;
                f.convertTo(image, CV_16UC1, 1000.0);
            } else {
                image = cv_bridge::toCvCopy(msg, msg->encoding)->image;
            }
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "深度图转换失败: %s", e.what());
            return;
        }
        if (image.empty()) return;
        const auto stamp_ns = StampToNanoseconds(msg->header.stamp);
        {
            const std::lock_guard<std::mutex> lock{mutex_};
            depth_history_.push_back({stamp_ns, image.clone()});
            PruneHistory(&depth_history_, max_history_size_);
        }
        TryProcessFrame(stamp_ns);
    }

    void SetCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr &msg) {
        if (!camera_info_received_) {
            estimator_->updateIntrinsics(msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
            if (estimator_->loadModel(obj_path_, m3t_dir_)) {
                RCLCPP_INFO(this->get_logger(), "多目标 M3T 模型装载成功！");
            }
            camera_info_received_ = true;
        }
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

        std_msgs::msg::Header header;
        header.stamp.sec = static_cast<int32_t>(color->stamp_ns / 1000000000LL);
        header.stamp.nanosec = static_cast<uint32_t>(color->stamp_ns % 1000000000LL);
        header.frame_id = "camera_color_optical_frame";

        processFrame(color->image, depth->image, header);
    }

    void processFrame(const cv::Mat &color, const cv::Mat &depth,
                      const std_msgs::msg::Header &header) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. YOLO 检测
        auto detections = detector_->detect(color, depth);
        auto t_detect = std::chrono::high_resolution_clock::now();

        cv::Mat display = color.clone();
        detector_->drawDetections(display, detections);

        // 🔥 2. 多目标 M3T 求解主接口
        auto valid_poses = estimator_->estimateMulti(detections, depth, color);

        // 3. 遍历并发布所有目标
        for (size_t i = 0; i < valid_poses.size(); ++i) {
            const auto& pose = valid_poses[i];

            estimator_->drawPoseAxes(display, pose);
            publishPose(pose, header, i);

            Transform T_cam_obj = ArmTransform::fromMatrix(pose.T_cam_obj);
            Transform T_ee_obj = arm_transform_->computeEEToObject(T_cam_obj);
            publishArmPose(T_ee_obj, header);
            broadcastTF(pose, header, i); // 广播: energy_unit_0, energy_unit_1, energy_unit_2
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double detect_ms = std::chrono::duration<double, std::milli>(t_detect - t_start).count();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        char info[256];
        snprintf(info, sizeof(info),
                 "Det: %.1fms | Total: %.1fms | YOLO: %zu | Tracked: %zu",
                 detect_ms, total_ms, detections.size(), valid_poses.size());
        cv::putText(display, info, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        result_image_pub_->publish(*cv_bridge::CvImage(header, "bgr8", display).toImageMsg());

        cv::imshow("Energy Unit", display);
        cv::waitKey(1);

        frame_count_++;
        if (frame_count_ % save_interval_ == 0) {
            char p[512];
            snprintf(p, sizeof(p), "%s/images/frame_%06d.jpg", result_dir_.c_str(), frame_count_);
            cv::imwrite(p, display);
        }

        if (!valid_poses.empty()) {
            uint64_t stamp_ns = header.stamp.sec * 1000000000ULL + header.stamp.nanosec;
            for (size_t i = 0; i < valid_poses.size(); ++i) {
                const auto &p = valid_poses[i];
                pose_csv_ << frame_count_ << "," << stamp_ns << "," << i << ","
                          << p.translation.x() << "," << p.translation.y() << "," << p.translation.z() << ","
                          << p.quaternion.w() << "," << p.quaternion.x() << "," << p.quaternion.y() << "," << p.quaternion.z() << ","
                          << p.euler_angles.x() << "," << p.euler_angles.y() << "," << p.euler_angles.z() << ","
                          << detect_ms << "," << total_ms << "," << detections.size() << "\n";
            }
            pose_csv_.flush();
        }
    }

    void publishPose(const PoseResult &p, const std_msgs::msg::Header &h, size_t i) {
        auto m = std::make_shared<geometry_msgs::msg::PoseStamped>();
        m->header = h; m->header.frame_id = "camera_link";
        m->pose.position.x = p.translation.x(); m->pose.position.y = p.translation.y(); m->pose.position.z = p.translation.z();
        m->pose.orientation.w = p.quaternion.w(); m->pose.orientation.x = p.quaternion.x();
        m->pose.orientation.y = p.quaternion.y(); m->pose.orientation.z = p.quaternion.z();
        if (i == 0) pose_pub_->publish(*m);
    }

    void publishArmPose(const Transform &T, const std_msgs::msg::Header &h) {
        auto m = std::make_shared<geometry_msgs::msg::PoseStamped>();
        m->header = h; m->header.frame_id = "arm_end_effector";
        m->pose.position.x = T.translation.x(); m->pose.position.y = T.translation.y(); m->pose.position.z = T.translation.z();
        m->pose.orientation.w = T.rotation.w(); m->pose.orientation.x = T.rotation.x();
        m->pose.orientation.y = T.rotation.y(); m->pose.orientation.z = T.rotation.z();
        arm_pose_pub_->publish(*m);
    }

    void broadcastTF(const PoseResult &p, const std_msgs::msg::Header &h, size_t i) {
        geometry_msgs::msg::TransformStamped t;
        t.header = h; t.header.frame_id = "camera_link";
        t.child_frame_id = "energy_unit_" + std::to_string(i);
        t.transform.translation.x = p.translation.x(); t.transform.translation.y = p.translation.y(); t.transform.translation.z = p.translation.z();
        t.transform.rotation.w = p.quaternion.w(); t.transform.rotation.x = p.quaternion.x();
        t.transform.rotation.y = p.quaternion.y(); t.transform.rotation.z = p.quaternion.z();
        tf_broadcaster_->sendTransform(t);
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