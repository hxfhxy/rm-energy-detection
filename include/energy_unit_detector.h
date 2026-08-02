#pragma once
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace energy_unit {

struct Detection {
    cv::RotatedRect bbox;
    cv::Rect bbox_axis_aligned;
    float confidence;
    cv::Point2f center;
    float depth_value;
    int label;
    cv::Mat mask; 
};

struct DetectorConfig {
    std::string model_path = "model/yolo/best_seg.onnx";
    std::string device = "CPU";
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    int input_size = 640;
};

class EnergyUnitDetector {
public:
    explicit EnergyUnitDetector(const DetectorConfig& config = DetectorConfig());
    std::vector<Detection> detect(const cv::Mat& color_bgr, const cv::Mat& depth);
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const;

private:
    DetectorConfig config_;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    bool is_seg_ = false;

    std::vector<Detection> runInference(const cv::Mat& color_bgr, const cv::Mat& depth);
    float getDepth(const cv::Mat& depth, const cv::Point2f& center) const;
    std::vector<int> nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float threshold);
    cv::Mat decodeMask(const float* mask_coeffs, const float* proto_data, int proto_h, int proto_w, const cv::Rect& bbox, int img_w, int img_h);
};

}  // namespace energy_unit