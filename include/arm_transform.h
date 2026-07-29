#pragma once
/**
 * @file arm_transform.h
 * @brief 机械臂末端到能量单元的坐标变换
 *
 * 定义 hand-eye 标定矩阵，将相机坐标系下的位姿
 * 变换到机械臂基座坐标系，用于抓取控制。
 */

#include <Eigen/Dense>
#include <string>

namespace energy_unit {

/// 齐次变换矩阵封装
struct Transform {
    Eigen::Matrix4d matrix;      ///< 4x4 齐次变换
    Eigen::Quaterniond rotation; ///< 四元数表示
    Eigen::Vector3d translation; ///< 平移向量

    Transform() : matrix(Eigen::Matrix4d::Identity()),
                  rotation(Eigen::Quaterniond::Identity()),
                  translation(Eigen::Vector3d::Zero()) {}
};

class ArmTransform {
public:
    /**
     * @brief 构造函数
     * @param T_base_ee    基座到末端的变换 (可从 TF 树获取)
     * @param T_ee_cam     末端到相机的变换 (hand-eye 标定结果)
     */
    ArmTransform(const Transform& T_base_ee = Transform(),
                 const Transform& T_ee_cam = Transform());

    /**
     * @brief 设置基座到末端的变换
     */
    void setBaseToEE(const Transform& T);

    /**
     * @brief 设置末端到相机的变换（hand-eye 标定）
     */
    void setEEToCamera(const Transform& T);

    /**
     * @brief 计算基座到能量单元的完整变换链
     *
     * T_base_obj = T_base_ee * T_ee_cam * T_cam_obj
     *
     * @param T_cam_obj  相机到能量单元的变换（来自 PoseEstimator）
     * @return 基座到能量单元的变换
     */
    Transform computeBaseToObject(const Transform& T_cam_obj) const;

    /**
     * @brief 计算末端到能量单元的变换
     *
     * T_ee_obj = T_ee_cam * T_cam_obj
     *
     * @param T_cam_obj  相机到能量单元的变换
     * @return 末端到能量单元的变换
     */
    Transform computeEEToObject(const Transform& T_cam_obj) const;

    /**
     * @brief 将 PoseResult 的 4x4 矩阵转为 Transform
     */
    static Transform fromMatrix(const Eigen::Matrix4d& mat);

    /**
     * @brief 从欧拉角 (度) + 平移构建 Transform
     */
    static Transform fromEuler(double roll_deg, double pitch_deg,
                               double yaw_deg,
                               double tx, double ty, double tz);

private:
    Transform T_base_ee_;   ///< 基座到末端
    Transform T_ee_cam_;    ///< 末端到相机
};

}  // namespace energy_unit
