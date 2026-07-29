#include "energy_unit_detector.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace energy_unit {

EnergyUnitDetector::EnergyUnitDetector(const DetectorConfig& config)
    : config_(config) {
    printf("[Detector] 加载模型: %s\n", config_.model_path.c_str());

    // 1. 读取模型
    auto model = core_.read_model(config_.model_path);

    // 2. 预处理设置
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_shape({1, config_.input_size, config_.input_size, 3})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255.0f, 255.0f, 255.0f});
    ppp.input().model().set_layout("NCHW");
    model = ppp.build();

    // 3. 编译
    compiled_model_ = core_.compile_model(model, config_.device);
    infer_request_ = compiled_model_.create_infer_request();

    // 4. 检测输出格式
    auto out_shape = compiled_model_.output(0).get_shape();
    int channels = (int)out_shape[1];
    if (channels > 50) {
        int tmp = channels; channels = (int)out_shape[2]; (void)tmp;
    }
    is_seg_ = (channels > 5);  // 5=检测, 37+=seg

    fprintf(stderr, "[Detector] OpenVINO: device=%s, mode=%s\n",
            config_.device.c_str(), is_seg_ ? "SEG" : "DET");
}

std::vector<Detection> EnergyUnitDetector::detect(const cv::Mat& color_bgr,
                                                   const cv::Mat& depth) {
    auto detections = runInference(color_bgr, depth);
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    return detections;
}

std::vector<Detection> EnergyUnitDetector::runInference(
    const cv::Mat& color_bgr, const cv::Mat& depth) {

    std::vector<Detection> results;
    int img_h = color_bgr.rows;
    int img_w = color_bgr.cols;

    // 1. 预处理：resize
    cv::Mat resized;
    cv::resize(color_bgr, resized,
               cv::Size(config_.input_size, config_.input_size));

    // 2. 创建输入 tensor 并推理
    ov::Tensor input_tensor(compiled_model_.input().get_element_type(),
                            compiled_model_.input().get_shape(),
                            resized.data);
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();

    ov::Tensor det_tensor;
    ov::Tensor proto_tensor;
    for (size_t i = 0; i < compiled_model_.outputs().size(); ++i) {
        auto tensor = infer_request_.get_output_tensor(i);
        if (tensor.get_shape().size() == 3) det_tensor = tensor;
        else if (tensor.get_shape().size() == 4) proto_tensor = tensor;
    }

    if (!det_tensor) return results;

    const float* det_data = det_tensor.data<float>();
    ov::Shape det_shape = det_tensor.get_shape();

    int num_features = 0, num_anchors = 0;
    if (det_shape.size() == 3) {
        int d1 = (int)det_shape[1], d2 = (int)det_shape[2];
        if (d1 <= 50 && d2 > 50) { num_features = d1; num_anchors = d2; }
        else if (d2 <= 50 && d1 > 50) { num_anchors = d1; num_features = d2; }
        else return results;
    }
    if (num_anchors == 0) return results;

    // =========================================================================
    // 🔥 解决报错的核心点：在此处提前声明并获取 proto_data、proto_h、proto_w
    // =========================================================================
    const float* proto_data = nullptr;
    int proto_h = 0, proto_w = 0;
    if (is_seg_ && proto_tensor) {
        proto_data = proto_tensor.data<float>();
        auto ps = proto_tensor.get_shape();
        proto_h = (int)ps[2];
        proto_w = (int)ps[3];
    }

    // 3. 计算通道布局
    int num_mask_coeffs = is_seg_ ? 32 : 0;
    int num_classes = num_features - 4 - num_mask_coeffs;

    const float* cx_ptr = det_data;
    const float* cy_ptr = det_data + num_anchors;
    const float* w_ptr  = det_data + 2 * num_anchors;
    const float* h_ptr  = det_data + 3 * num_anchors;

    float sx = (float)img_w / config_.input_size;
    float sy = (float)img_h / config_.input_size;

    // 4. 收集候选框
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> anchor_indices;

    for (int i = 0; i < num_anchors; ++i) {
        float max_conf = 0.0f;
        int best_class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float conf = det_data[(4 + c) * num_anchors + i];
            if (conf > max_conf) {
                max_conf = conf;
                best_class_id = c;
            }
        }

        if (max_conf < config_.conf_threshold) continue;

        float cx = cx_ptr[i] * sx;
        float cy = cy_ptr[i] * sy;
        float bw = w_ptr[i]  * sx;
        float bh = h_ptr[i]  * sy;

        int x1 = std::max(0, (int)(cx - bw / 2));
        int y1 = std::max(0, (int)(cy - bh / 2));
        int x2 = std::min(img_w, (int)(cx + bw / 2));
        int y2 = std::min(img_h, (int)(cy + bh / 2));

        int w = x2 - x1;
        int h = y2 - y1;

        if (w <= 0 || h <= 0) continue;

        boxes.emplace_back(x1, y1, w, h);
        confidences.push_back(max_conf);
        class_ids.push_back(best_class_id);
        anchor_indices.push_back(i);
    }

    // 5. NMS 过滤
    auto keep = nms(boxes, confidences, config_.nms_threshold);

    // 6. 构建最终检测结果
    for (int idx : keep) {
        Detection det;
        det.bbox_axis_aligned = boxes[idx];
        det.center = cv::Point2f(boxes[idx].x + boxes[idx].width / 2.0f,
                                 boxes[idx].y + boxes[idx].height / 2.0f);
        det.bbox = cv::RotatedRect(det.center,
                                   cv::Size2f(boxes[idx].width,
                                              boxes[idx].height), 0.0f);
        det.confidence = confidences[idx];
        det.label = class_ids[idx];
        det.depth_value = getDepth(depth, det.center);

        // 解码 Mask (此时 proto_data, proto_h, proto_w 在全函数作用域均有效)
        if (is_seg_ && proto_data) {
            float mask_coeffs[32];
            int anchor_i = anchor_indices[idx];
            for (int k = 0; k < num_mask_coeffs; ++k) {
                mask_coeffs[k] = det_data[(4 + num_classes + k) * num_anchors + anchor_i];
            }
            det.mask = decodeMask(mask_coeffs, proto_data,
                                  proto_h, proto_w,
                                  boxes[idx], img_w, img_h);
        }

        results.push_back(det);
    }

    return results;
}

// ================================================================
// Mask 解码（参考用户提供的 OpenVINO 实现）
// ================================================================

cv::Mat EnergyUnitDetector::decodeMask(
    const float* mask_coeffs, const float* proto_data,
    int proto_h, int proto_w,
    const cv::Rect& bbox, int img_w, int img_h) {

    // 1. mask = sigmoid( sum(proto[k] * coeff[k]) )
    cv::Mat mask(proto_h, proto_w, CV_32F, cv::Scalar(0.0f));
    for (int k = 0; k < 32; ++k) {
        float coeff = mask_coeffs[k];
        const float* proto_ch = proto_data + k * proto_h * proto_w;
        for (int y = 0; y < proto_h; ++y) {
            float* row = mask.ptr<float>(y);
            for (int x = 0; x < proto_w; ++x) {
                row[x] += proto_ch[y * proto_w + x] * coeff;
            }
        }
    }
    // sigmoid
    for (int y = 0; y < proto_h; ++y) {
        float* row = mask.ptr<float>(y);
        for (int x = 0; x < proto_w; ++x) {
            row[x] = 1.0f / (1.0f + std::exp(-row[x]));
        }
    }

    // 2. 裁剪 bbox 到有效范围
    int x1 = std::max(0, bbox.x);
    int y1 = std::max(0, bbox.y);
    int x2 = std::min(img_w, bbox.x + bbox.width);
    int y2 = std::min(img_h, bbox.y + bbox.height);
    if (x2 <= x1 || y2 <= y1) return cv::Mat();

    // 3. bbox → proto 坐标
    float sx = (float)proto_w / (float)config_.input_size;
    float sy = (float)proto_h / (float)config_.input_size;

    int px1 = std::max(0, std::min(proto_w - 1,
             (int)((float)x1 / (float)img_w * config_.input_size * sx)));
    int py1 = std::max(0, std::min(proto_h - 1,
             (int)((float)y1 / (float)img_h * config_.input_size * sy)));
    int px2 = std::max(1, std::min(proto_w,
             (int)((float)x2 / (float)img_w * config_.input_size * sx)));
    int py2 = std::max(1, std::min(proto_h,
             (int)((float)y2 / (float)img_h * config_.input_size * sy)));
    if (px2 <= px1 || py2 <= py1) return cv::Mat();

    // 4. 裁剪并 resize 到 bbox 尺寸
    cv::Mat cropped = mask(cv::Rect(px1, py1, px2 - px1, py2 - py1)).clone();
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(x2 - x1, y2 - y1));

    // 5. 二值化 + 形态学
    cv::Mat binary;
    cv::threshold(resized, binary, 0.5, 255, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::dilate(binary, binary, kernel);

    return binary;
}

// ================================================================
// NMS
// ================================================================

std::vector<int> EnergyUnitDetector::nms(const std::vector<cv::Rect>& boxes,
                                          const std::vector<float>& scores,
                                          float threshold) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int best = indices.front();
        keep.push_back(best);
        indices.erase(indices.begin());

        std::vector<int> remaining;
        for (int idx : indices) {
            int ix1 = std::max(boxes[best].x, boxes[idx].x);
            int iy1 = std::max(boxes[best].y, boxes[idx].y);
            int ix2 = std::min(boxes[best].x + boxes[best].width,
                               boxes[idx].x + boxes[idx].width);
            int iy2 = std::min(boxes[best].y + boxes[best].height,
                               boxes[idx].y + boxes[idx].height);
            float inter = std::max(0, ix2 - ix1) * std::max(0, iy2 - iy1);
            float union_area = boxes[best].area() + boxes[idx].area() - inter;
            float iou = (union_area > 0) ? inter / union_area : 0;
            if (iou <= threshold) remaining.push_back(idx);
        }
        indices = remaining;
    }
    return keep;
}

// ================================================================
// 深度采样
// ================================================================

float EnergyUnitDetector::getDepth(const cv::Mat& depth,
                                    const cv::Point2f& center) const {
    int cx = static_cast<int>(center.x);
    int cy = static_cast<int>(center.y);
    if (cx < 0 || cy < 0 || cx >= depth.cols || cy >= depth.rows) return 0.0f;

    int half = 2;
    std::vector<uint16_t> values;
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            int x = cx + dx, y = cy + dy;
            if (x >= 0 && y >= 0 && x < depth.cols && y < depth.rows) {
                uint16_t v = depth.at<uint16_t>(y, x);
                if (v > 0 && v < 10000) values.push_back(v);
            }
        }
    }
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    return static_cast<float>(values[values.size() / 2]);
}

// ================================================================
// 可视化
// ================================================================

void EnergyUnitDetector::drawDetections(
    cv::Mat& image,
    const std::vector<Detection>& detections) const {

    for (const auto& det : detections) {
        // 根据不同 class_id 区分颜色（例如 Class 0 红，Class 1 绿）
        cv::Scalar color = (det.label == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        // seg: 半透明 mask 叠加
        if (!det.mask.empty()) {
            cv::Rect roi = det.bbox_axis_aligned & cv::Rect(0, 0, image.cols, image.rows);
            if (roi.width > 0 && roi.height > 0 &&
                det.mask.cols == roi.width && det.mask.rows == roi.height) {
                cv::Mat image_roi = image(roi);
                for (int y = 0; y < roi.height; ++y) {
                    for (int x = 0; x < roi.width; ++x) {
                        if (det.mask.at<uchar>(y, x) > 0) {
                            auto& px = image_roi.at<cv::Vec3b>(y, x);
                            px[0] = px[0] * 0.7 + color[0] * 0.3;
                            px[1] = px[1] * 0.7 + color[1] * 0.3;
                            px[2] = px[2] * 0.7 + color[2] * 0.3;
                        }
                    }
                }
            }
        }

        cv::rectangle(image, det.bbox_axis_aligned, color, 2);
        cv::circle(image, det.center, 4, color, -1);

        char text[128];
        // 动态显示 class_id
        snprintf(text, sizeof(text), "cls:%d %.0f%% d=%.0fmm",
                 det.label, det.confidence * 100.0f, det.depth_value);
        cv::putText(image, text,
                    cv::Point(det.bbox_axis_aligned.x,
                              det.bbox_axis_aligned.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

}  // namespace energy_unit