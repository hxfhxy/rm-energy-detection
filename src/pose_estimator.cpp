#include "pose_estimator.h"
#include "energy_unit_detector.h"
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>

namespace energy_unit {

// ==================== 辅助工具函数 ====================
namespace {
    // 简单k-d树节点（用于最近邻搜索，点数不多时暴力亦可，但为效率实现基本k-d树）
    struct SimpleKDNode {
        Eigen::Vector3d pt;
        int axis;
        SimpleKDNode* left = nullptr;
        SimpleKDNode* right = nullptr;
    };

    SimpleKDNode* buildKDTree(std::vector<Eigen::Vector3d> pts, int depth = 0) {
        if (pts.empty()) return nullptr;
        int axis = depth % 3;
        size_t mid = pts.size() / 2;
        std::nth_element(pts.begin(), pts.begin() + mid, pts.end(),
            [axis](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a[axis] < b[axis];
            });
        SimpleKDNode* node = new SimpleKDNode{pts[mid], axis, nullptr, nullptr};
        node->left = buildKDTree({pts.begin(), pts.begin() + mid}, depth + 1);
        node->right = buildKDTree({pts.begin() + mid + 1, pts.end()}, depth + 1);
        return node;
    }

    void deleteKDTree(SimpleKDNode* node) {
        if (!node) return;
        deleteKDTree(node->left);
        deleteKDTree(node->right);
        delete node;
    }

    void nearestNeighbor(SimpleKDNode* node, const Eigen::Vector3d& query,
                         double& best_dist, Eigen::Vector3d& best_pt) {
        if (!node) return;
        double d = (node->pt - query).norm();
        if (d < best_dist) {
            best_dist = d;
            best_pt = node->pt;
        }
        int axis = node->axis;
        double diff = query[axis] - node->pt[axis];
        SimpleKDNode* first = diff <= 0 ? node->left : node->right;
        SimpleKDNode* second = diff <= 0 ? node->right : node->left;
        nearestNeighbor(first, query, best_dist, best_pt);
        if (std::abs(diff) < best_dist) {
            nearestNeighbor(second, query, best_dist, best_pt);
        }
    }

    // 欧式聚类（使用简单的区域生长 + 空间网格）
    std::vector<std::vector<Eigen::Vector3d>> euclideanCluster(
        const std::vector<Eigen::Vector3d>& pts, double tolerance, int min_size) {
        // 构建空间网格加速邻域查询
        auto [min_x, max_x] = std::minmax_element(pts.begin(), pts.end(),
            [](const auto& a, const auto& b) { return a.x() < b.x(); });
        auto [min_y, max_y] = std::minmax_element(pts.begin(), pts.end(),
            [](const auto& a, const auto& b) { return a.y() < b.y(); });
        double grid_size = tolerance;
        int nx = std::max(1, (int)((max_x->x() - min_x->x()) / grid_size) + 1);
        int ny = std::max(1, (int)((max_y->y() - min_y->y()) / grid_size) + 1);
        std::vector<std::vector<int>> grid(nx * ny);
        for (size_t i = 0; i < pts.size(); ++i) {
            int gx = (int)((pts[i].x() - min_x->x()) / grid_size);
            int gy = (int)((pts[i].y() - min_y->y()) / grid_size);
            gx = std::clamp(gx, 0, nx - 1);
            gy = std::clamp(gy, 0, ny - 1);
            grid[gy * nx + gx].push_back((int)i);
        }

        std::vector<bool> visited(pts.size(), false);
        std::vector<std::vector<Eigen::Vector3d>> clusters;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (visited[i]) continue;
            std::vector<Eigen::Vector3d> cluster;
            std::vector<int> stack = {(int)i};
            visited[i] = true;
            while (!stack.empty()) {
                int idx = stack.back();
                stack.pop_back();
                cluster.push_back(pts[idx]);
                int gx = (int)((pts[idx].x() - min_x->x()) / grid_size);
                int gy = (int)((pts[idx].y() - min_y->y()) / grid_size);
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nxg = gx + dx, nyg = gy + dy;
                        if (nxg < 0 || nxg >= nx || nyg < 0 || nyg >= ny) continue;
                        for (int nbr : grid[nyg * nx + nxg]) {
                            if (visited[nbr]) continue;
                            if ((pts[idx] - pts[nbr]).norm() <= tolerance) {
                                visited[nbr] = true;
                                stack.push_back(nbr);
                            }
                        }
                    }
                }
            }
            if ((int)cluster.size() >= min_size)
                clusters.push_back(cluster);
        }
        return clusters;
    }
}

// ==================== PoseEstimator 实现 ====================
PoseEstimator::PoseEstimator(const EstimatorConfig& config) : config_(config) {
    camera_matrix_ = (cv::Mat_<double>(3, 3) << config_.fx, 0, config_.cx, 0, config_.fy, config_.cy, 0, 0, 1);
    dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
    buildModelCloud();  // 生成模型点云
}

void PoseEstimator::updateIntrinsics(double fx, double fy, double cx, double cy) {
    config_.fx = fx; config_.fy = fy; config_.cx = cx; config_.cy = cy;
    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

void PoseEstimator::buildModelCloud() {
    double r = config_.cylinder_radius;
    double h = config_.cylinder_height;
    model_cloud_.clear();
    // 生成圆柱体点云：侧面 + 顶底圆盘
    int n_side = 200;    // 圆周采样
    int n_h = 20;        // 高度采样
    for (int i = 0; i < n_side; ++i) {
        double angle = 2 * M_PI * i / n_side;
        double x = r * cos(angle);
        double y = r * sin(angle);
        for (int j = 0; j <= n_h; ++j) {
            double z = -h/2 + h * j / n_h;
            model_cloud_.emplace_back(x, y, z);
        }
    }
    // 顶底面
    int n_disk = 100;
    for (int i = 0; i < n_disk; ++i) {
        double angle = 2 * M_PI * i / n_disk;
        for (double rad : {r * 0.7, r * 0.85, r}) {
            double x = rad * cos(angle);
            double y = rad * sin(angle);
            model_cloud_.emplace_back(x, y, -h/2);
            model_cloud_.emplace_back(x, y, h/2);
        }
    }
    // 可进一步添加内部特征点，但对称物体ICP主要靠侧面
}

std::vector<Eigen::Vector3d> PoseEstimator::extractPointCloud(const Detection& det, const cv::Mat& depth) const {
    std::vector<Eigen::Vector3d> points;
    cv::Rect roi = det.bbox_axis_aligned & cv::Rect(0, 0, depth.cols, depth.rows);
    if (roi.width <= 0 || roi.height <= 0) return points;

    const bool has_mask = !det.mask.empty();
    // 计算掩码内深度中值作为门控
    std::vector<double> depths;
    for (int y = roi.y; y < roi.y + roi.height; y++) {
        const uint16_t* drow = depth.ptr<uint16_t>(y);
        const uchar* mrow = has_mask ? det.mask.ptr<uchar>(y - roi.y) : nullptr;
        for (int x = roi.x; x < roi.x + roi.width; x++) {
            if (has_mask && mrow[x - roi.x] == 0) continue;
            uint16_t d = drow[x];
            if (d > 100 && d < 5000) depths.push_back(d / 1000.0);
        }
    }
    if (depths.empty()) return points;
    std::nth_element(depths.begin(), depths.begin() + depths.size()/2, depths.end());
    double median_depth = depths[depths.size()/2];

    // 提取点云，并过滤与中值深度差超过阈值的点（背景剔除）
    for (int y = roi.y; y < roi.y + roi.height; y++) {
        const uint16_t* drow = depth.ptr<uint16_t>(y);
        const uchar* mrow = has_mask ? det.mask.ptr<uchar>(y - roi.y) : nullptr;
        for (int x = roi.x; x < roi.x + roi.width; x++) {
            if (has_mask && mrow[x - roi.x] == 0) continue;
            uint16_t d = drow[x];
            if (d > 100 && d < 5000) {
                double z = d / 1000.0;
                if (std::abs(z - median_depth) > 0.10) continue; // 10cm 门控
                double x_m = (x - config_.cx) * z / config_.fx;
                double y_m = (y - config_.cy) * z / config_.fy;
                points.emplace_back(x_m, y_m, z);
            }
        }
    }
    return points;
}

std::vector<Eigen::Vector3d> PoseEstimator::voxelDownsample(const std::vector<Eigen::Vector3d>& pts, double leaf) const {
    if (pts.empty()) return pts;
    // 简单格网滤波
    auto [min_x, max_x] = std::minmax_element(pts.begin(), pts.end(),
        [](const auto& a, const auto& b) { return a.x() < b.x(); });
    auto [min_y, max_y] = std::minmax_element(pts.begin(), pts.end(),
        [](const auto& a, const auto& b) { return a.y() < b.y(); });
    auto [min_z, max_z] = std::minmax_element(pts.begin(), pts.end(),
        [](const auto& a, const auto& b) { return a.z() < b.z(); });
    int nx = std::max(1, (int)((max_x->x() - min_x->x())/leaf) + 1);
    int ny = std::max(1, (int)((max_y->y() - min_y->y())/leaf) + 1);
    int nz = std::max(1, (int)((max_z->z() - min_z->z())/leaf) + 1);
    std::vector<std::vector<std::vector<int>>> bins(nx, std::vector<std::vector<int>>(ny, std::vector<int>(nz, -1)));
    std::vector<Eigen::Vector3d> filtered;
    for (size_t i = 0; i < pts.size(); ++i) {
        int gx = std::clamp((int)((pts[i].x() - min_x->x())/leaf), 0, nx-1);
        int gy = std::clamp((int)((pts[i].y() - min_y->y())/leaf), 0, ny-1);
        int gz = std::clamp((int)((pts[i].z() - min_z->z())/leaf), 0, nz-1);
        if (bins[gx][gy][gz] == -1) {
            bins[gx][gy][gz] = (int)filtered.size();
            filtered.push_back(pts[i]);
        }
    }
    return filtered;
}

std::vector<Eigen::Vector3d> PoseEstimator::statisticalOutlierRemoval(const std::vector<Eigen::Vector3d>& pts, int k, double std_mul) const {
    if (pts.size() < (size_t)k) return pts;
    // 为每个点计算平均距离
    std::vector<double> mean_dists(pts.size());
    // 使用暴力搜索，点数少时可行
    for (size_t i = 0; i < pts.size(); ++i) {
        std::vector<double> dists;
        for (size_t j = 0; j < pts.size(); ++j) {
            if (i == j) continue;
            dists.push_back((pts[i] - pts[j]).norm());
        }
        std::nth_element(dists.begin(), dists.begin() + k, dists.end());
        double sum = 0;
        for (int m = 0; m < k; ++m) sum += dists[m];
        mean_dists[i] = sum / k;
    }
    double sum = 0;
    for (double d : mean_dists) sum += d;
    double mean = sum / pts.size();
    double sq_sum = 0;
    for (double d : mean_dists) sq_sum += (d - mean) * (d - mean);
    double stddev = std::sqrt(sq_sum / pts.size());
    double threshold = mean + std_mul * stddev;

    std::vector<Eigen::Vector3d> inliers;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (mean_dists[i] < threshold)
            inliers.push_back(pts[i]);
    }
    return inliers;
}

std::vector<Eigen::Vector3d> PoseEstimator::euclideanClusterSelect(const std::vector<Eigen::Vector3d>& pts, double tolerance, int min_size) const {
    auto clusters = euclideanCluster(pts, tolerance, min_size);
    if (clusters.empty()) return pts;   // 失败时返回原样
    // 选择点数最多的簇
    return *std::max_element(clusters.begin(), clusters.end(),
        [](const auto& a, const auto& b) { return a.size() < b.size(); });
}

Eigen::Vector3d PoseEstimator::computePCALongAxis(const std::vector<Eigen::Vector3d>& pts, Eigen::Vector3d& centroid) const {
    centroid = Eigen::Vector3d::Zero();
    for (auto& p : pts) centroid += p;
    centroid /= pts.size();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (auto& p : pts) {
        Eigen::Vector3d d = p - centroid;
        cov += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    // 长轴对应最大特征值的特征向量（col(2)）
    return solver.eigenvectors().col(2).normalized();
}

std::vector<Eigen::Matrix4d> PoseEstimator::generateInitialCandidates(
    const Eigen::Vector3d& centroid,
    const Eigen::Vector3d& long_axis,
    int num_rolls) const {
    
    std::vector<Eigen::Matrix4d> candidates;
    // 两个极性
    Eigen::Vector3d axes[2] = {long_axis, -long_axis};
    for (int p = 0; p < 2; ++p) {
        // 构建基础旋转：使物体Z轴与 long_axis 对齐
        Eigen::Matrix3d R_base = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), axes[p]).toRotationMatrix();
        // 绕 long_axis 均匀采样 roll 角
        for (int i = 0; i < num_rolls; ++i) {
            double roll = 2 * M_PI * i / num_rolls;
            Eigen::Matrix3d R_roll = Eigen::AngleAxisd(roll, axes[p]) * R_base;
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = R_roll;
            // 平移初值：模型中心对齐到场景点云质心
            T.block<3,1>(0,3) = centroid - R_roll * Eigen::Vector3d(0, 0, 0); // 模型中心在物体系原点
            candidates.push_back(T);
        }
    }
    return candidates;
}

PoseEstimator::ICPResult PoseEstimator::trimmedICP(
    const std::vector<Eigen::Vector3d>& source, // 模型点云
    const std::vector<Eigen::Vector3d>& target, // 场景点云
    const Eigen::Matrix4d& init_T,
    double max_dist,
    int max_iter,
    double trim_ratio) const {

    // 构建目标点云的k-d树
    SimpleKDNode* kd_root = buildKDTree(target);

    Eigen::Matrix4d T = init_T;
    double trimmed_rmse = std::numeric_limits<double>::max();
    double corr_ratio = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // 变换源点云
        std::vector<Eigen::Vector3d> src_transformed(source.size());
        for (size_t i = 0; i < source.size(); ++i) {
            Eigen::Vector4d h(source[i].x(), source[i].y(), source[i].z(), 1.0);
            h = T * h;
            src_transformed[i] = h.head<3>();
        }

        // 寻找对应点并计算残差
        std::vector<std::pair<int, Eigen::Vector3d>> correspondences; // (src_idx, target_pt)
        std::vector<double> residuals;
        for (size_t i = 0; i < src_transformed.size(); ++i) {
            double best_dist = max_dist;
            Eigen::Vector3d best_pt(0,0,0);
            nearestNeighbor(kd_root, src_transformed[i], best_dist, best_pt);
            if (best_dist < max_dist) {
                correspondences.emplace_back(i, best_pt);
                residuals.push_back(best_dist);
            }
        }

        if (correspondences.size() < 10) break;

        // Trim: 按残差排序，保留前 trim_ratio 比例的点
        std::vector<int> idx(residuals.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return residuals[a] < residuals[b]; });
        size_t keep_num = std::max((size_t)10, (size_t)(correspondences.size() * trim_ratio));
        std::vector<Eigen::Vector3d> src_pts, tgt_pts;
        for (size_t i = 0; i < keep_num; ++i) {
            int orig = idx[i];
            src_pts.push_back(src_transformed[correspondences[orig].first]);
            tgt_pts.push_back(correspondences[orig].second);
        }

        // 求解最小二乘刚体变换 (SVD)
        Eigen::Vector3d src_cent(0,0,0), tgt_cent(0,0,0);
        for (size_t i = 0; i < src_pts.size(); ++i) {
            src_cent += src_pts[i];
            tgt_cent += tgt_pts[i];
        }
        src_cent /= src_pts.size();
        tgt_cent /= src_pts.size();

        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < src_pts.size(); ++i) {
            H += (src_pts[i] - src_cent) * (tgt_pts[i] - tgt_cent).transpose();
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
        if (R.determinant() < 0) {
            Eigen::Matrix3d V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        Eigen::Vector3d t = tgt_cent - R * src_cent;
        Eigen::Matrix4d delta_T = Eigen::Matrix4d::Identity();
        delta_T.block<3,3>(0,0) = R;
        delta_T.block<3,1>(0,3) = t;

        T = delta_T * T;

        // 计算本次迭代后 trimmed RMSE
        double sq_sum = 0;
        for (size_t i = 0; i < keep_num; ++i) {
            sq_sum += (R * src_pts[i] + t - tgt_pts[i]).squaredNorm();
        }
        trimmed_rmse = std::sqrt(sq_sum / keep_num);
        corr_ratio = (double)correspondences.size() / source.size();
    }

    deleteKDTree(kd_root);
    return {T, trimmed_rmse, corr_ratio};
}

std::vector<PoseResult> PoseEstimator::estimateMulti(const std::vector<Detection>& detections,
                                                     const cv::Mat& depth,
                                                     const cv::Mat& /*color_bgr*/) {
    std::vector<PoseResult> results;

    struct DetData {
        Detection det;
        std::vector<Eigen::Vector3d> cloud;
        Eigen::Vector3d centroid;
        Eigen::Matrix4d pose;
        bool valid = false;
    };
    std::vector<DetData> current_dets;

    for (const auto& det : detections) {
        DetData data;
        data.det = det;

        // 1. 提取原始点云
        auto raw = extractPointCloud(det, depth);
        if (raw.size() < 30) {
            if(config_.verbose) printf("[Debug] 目标 %d 丢弃: 点云太少 (%zu)\n", det.label, raw.size());
            continue;
        }

        // 2. 滤波流水线
        auto filtered = voxelDownsample(raw, config_.voxel_leaf_size);
        filtered = statisticalOutlierRemoval(filtered, config_.stat_outlier_k, config_.stat_std_mul);
        filtered = euclideanClusterSelect(filtered, config_.cluster_tolerance, config_.cluster_min_size);
        if (filtered.empty()) continue;

        // 3. PCA 求长轴和质心
        Eigen::Vector3d centroid;
        Eigen::Vector3d long_axis = computePCALongAxis(filtered, centroid);

        // ========== 关键修改1：根据大头小头特征固定轴线方向 ==========
        {
            std::vector<double> projs;
            projs.reserve(filtered.size());
            for (const auto& p : filtered) {
                projs.push_back((p - centroid).dot(long_axis));
            }
            std::nth_element(projs.begin(), projs.begin() + projs.size()/2, projs.end());
            double median_proj = projs[projs.size()/2];

            std::vector<double> radii_pos, radii_neg;
            for (const auto& p : filtered) {
                double d_axis = (p - centroid).cross(long_axis).norm();
                double proj = (p - centroid).dot(long_axis);
                if (proj > median_proj) {
                    radii_pos.push_back(d_axis);
                } else {
                    radii_neg.push_back(d_axis);
                }
            }

            double avg_r_pos = 0, avg_r_neg = 0;
            if (!radii_pos.empty()) avg_r_pos = std::accumulate(radii_pos.begin(), radii_pos.end(), 0.0) / radii_pos.size();
            if (!radii_neg.empty()) avg_r_neg = std::accumulate(radii_neg.begin(), radii_neg.end(), 0.0) / radii_neg.size();

            if (avg_r_pos > avg_r_neg) {
                long_axis = -long_axis;   // 让 +Z 指向小头（底部）
            }
        }

        // 4. 候选生成
        auto candidates = generateInitialCandidates(centroid, long_axis, config_.num_roll_candidates);

        // ========== 关键修改2：利用投影 y 坐标过滤候选 ==========
        const double half_height = config_.cylinder_height / 2.0;
        const Eigen::Vector3d top_local(0, 0, half_height);
        const Eigen::Vector3d bot_local(0, 0, -half_height);
        std::vector<Eigen::Matrix4d> valid_candidates;

        for (const auto& T : candidates) {
            Eigen::Vector4d top_cam = T * Eigen::Vector4d(top_local.x(), top_local.y(), top_local.z(), 1.0);
            Eigen::Vector4d bot_cam = T * Eigen::Vector4d(bot_local.x(), bot_local.y(), bot_local.z(), 1.0);
            
            double v_top = (top_cam.y() * config_.fy / top_cam.z()) + config_.cy;
            double v_bot = (bot_cam.y() * config_.fy / bot_cam.z()) + config_.cy;
            
            if (v_top < v_bot) {
                valid_candidates.push_back(T);
            }
        }

        if (valid_candidates.empty()) {
            valid_candidates = candidates;
        }

        // 5. ICP 评估并选出最佳候选
        ICPResult best_res;
        best_res.trimmed_rmse = std::numeric_limits<double>::max();
        for (const auto& init : valid_candidates) {
            auto res = trimmedICP(model_cloud_, filtered, init,
                                  config_.icp_max_correspondence_dist,
                                  config_.icp_max_iter,
                                  config_.trimmed_ratio);
            if (res.trimmed_rmse < best_res.trimmed_rmse) {
                best_res = res;
            }
        }

        if (best_res.trimmed_rmse > 0.1 || best_res.correspondence_ratio < 0.3) {
            if(config_.verbose) printf("[Debug] 目标 %d ICP 失败: rmse=%.3f ratio=%.2f\n",
                                       det.label, best_res.trimmed_rmse, best_res.correspondence_ratio);
            continue;
        }

        data.valid = true;
        data.pose = best_res.T;
        data.centroid = centroid;
        current_dets.push_back(data);
    }

    // ========== 输出结果（新增 XY 轴固定） ==========
    for (size_t i = 0; i < current_dets.size(); ++i) {
        PoseResult res;
        // 固定绕轴转角：强制 X 轴水平向右，Y 轴由右手定则确定
        Eigen::Matrix3d R = current_dets[i].pose.block<3,3>(0,0);
        Eigen::Vector3d z_axis = R.col(2).normalized();                     // 保持 Z 轴不变
        Eigen::Vector3d x_axis = Eigen::Vector3d::UnitX() - z_axis * (z_axis.dot(Eigen::Vector3d::UnitX()));
        if (x_axis.norm() < 1e-6) {
            x_axis = Eigen::Vector3d::UnitY() - z_axis * (z_axis.dot(Eigen::Vector3d::UnitY()));
        }
        x_axis.normalize();
        Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d R_fixed;
        R_fixed.col(0) = x_axis;
        R_fixed.col(1) = y_axis;
        R_fixed.col(2) = z_axis;

        res.T_cam_obj = current_dets[i].pose;
        res.T_cam_obj.block<3,3>(0,0) = R_fixed;      // 替换旋转部分
        res.translation = current_dets[i].pose.block<3,1>(0,3);
        res.quaternion = Eigen::Quaterniond(R_fixed);
        res.axis_direction = z_axis;
        res.euler_angles = R_fixed.eulerAngles(2,1,0) * 180.0 / M_PI;
        res.label = current_dets[i].det.label;
        res.valid = true;
        results.push_back(res);
    }

    return results;
}


void PoseEstimator::drawPoseAxes(cv::Mat& image, const PoseResult& pose, double axis_length) const {
    if (!pose.valid) return;

    // 获取真实的圆柱物理半高 (例如 150mm / 2 = 75mm)
    double h_half = config_.cylinder_height / 2.0;

    std::vector<cv::Point3f> axis_pts_3d = {
        cv::Point3f(0, 0, 0),                           // 0: 质心原点
        cv::Point3f(axis_length, 0, 0),                 // 1: X 轴正向
        cv::Point3f(0, axis_length, 0),                 // 2: Y 轴正向
        cv::Point3f(0, 0, axis_length),                 // 3: Z 轴正向
        cv::Point3f(0, 0, -h_half),                     // 4: 真实的圆柱【底面】中心点
        cv::Point3f(0, 0, h_half)                       // 5: 真实的圆柱【顶面】中心点
    };

    cv::Mat rvec, tvec;
    Eigen::Matrix3d R = pose.T_cam_obj.block<3, 3>(0, 0);
    cv::Mat R_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R_cv.at<double>(i, j) = R(i, j);
    cv::Rodrigues(R_cv, rvec);

    tvec = (cv::Mat_<double>(3, 1) << pose.translation.x(), pose.translation.y(), pose.translation.z());

    // 3D 到 2D 图像重投影
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(axis_pts_3d, rvec, tvec, camera_matrix_, dist_coeffs_, img_pts);

    // ==========================================
    // 1. 画真实的物理圆柱中轴线 (黄色，带两端端点)
    // ==========================================
    cv::line(image, img_pts[4], img_pts[5], cv::Scalar(0, 255, 255), 3, cv::LINE_AA); // 黄色粗线
    cv::circle(image, img_pts[4], 5, cv::Scalar(0, 255, 255), -1); // 底端圆点
    cv::circle(image, img_pts[5], 5, cv::Scalar(0, 255, 255), -1); // 顶端圆点
    
    // 标注一下中轴线的头尾
    cv::putText(image, "Top", img_pts[5] + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    cv::putText(image, "Bot", img_pts[4] + cv::Point2f(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);

    // ==========================================
    // 2. 画原点和标准的 XYZ 坐标系
    // ==========================================
    cv::circle(image, img_pts[0], 4, cv::Scalar(255, 255, 255), -1); // 质心原点
    
    cv::arrowedLine(image, img_pts[0], img_pts[1], cv::Scalar(0, 0, 255), 2); // X 轴 (红)
    cv::arrowedLine(image, img_pts[0], img_pts[2], cv::Scalar(0, 255, 0), 2); // Y 轴 (绿)
    // Z 轴 (蓝) 盖在中轴线上，稍微细一点，体现方向
    cv::arrowedLine(image, img_pts[0], img_pts[3], cv::Scalar(255, 0, 0), 2); 

    // 标注 XYZ 字母
    cv::putText(image, "X", img_pts[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "Y", img_pts[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "Z", img_pts[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
}

}  // namespace energy_unit