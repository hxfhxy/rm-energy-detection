#include "arm_transform.h"
#include <cmath>

namespace energy_unit {

ArmTransform::ArmTransform(const Transform& T_base_ee, const Transform& T_ee_cam)
    : T_base_ee_(T_base_ee), T_ee_cam_(T_ee_cam) {}

void ArmTransform::setBaseToEE(const Transform& T) { T_base_ee_ = T; }
void ArmTransform::setEEToCamera(const Transform& T) { T_ee_cam_ = T; }

Transform ArmTransform::computeBaseToObject(const Transform& T_cam_obj) const {
    Transform result;
    result.matrix = T_base_ee_.matrix * T_ee_cam_.matrix * T_cam_obj.matrix;
    result.translation = result.matrix.block<3, 1>(0, 3);
    result.rotation = Eigen::Quaterniond(result.matrix.block<3, 3>(0, 0));
    result.rotation.normalize();
    return result;
}

Transform ArmTransform::computeEEToObject(const Transform& T_cam_obj) const {
    Transform result;
    result.matrix = T_ee_cam_.matrix * T_cam_obj.matrix;
    result.translation = result.matrix.block<3, 1>(0, 3);
    result.rotation = Eigen::Quaterniond(result.matrix.block<3, 3>(0, 0));
    result.rotation.normalize();
    return result;
}

Transform ArmTransform::fromMatrix(const Eigen::Matrix4d& mat) {
    Transform t;
    t.matrix = mat;
    t.translation = mat.block<3, 1>(0, 3);
    t.rotation = Eigen::Quaterniond(mat.block<3, 3>(0, 0));
    t.rotation.normalize();
    return t;
}

Transform ArmTransform::fromEuler(double roll_deg, double pitch_deg, double yaw_deg, double tx, double ty, double tz) {
    double r = roll_deg * M_PI / 180.0;
    double p = pitch_deg * M_PI / 180.0;
    double y = yaw_deg * M_PI / 180.0;
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX());
    Transform t;
    t.matrix = Eigen::Matrix4d::Identity();
    t.matrix.block<3, 3>(0, 0) = R;
    t.matrix(0, 3) = tx; t.matrix(1, 3) = ty; t.matrix(2, 3) = tz;
    t.translation = Eigen::Vector3d(tx, ty, tz);
    t.rotation = Eigen::Quaterniond(R);
    t.rotation.normalize();
    return t;
}

}  // namespace energy_unit