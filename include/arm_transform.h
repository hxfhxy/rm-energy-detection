#pragma once
#include <Eigen/Dense>
#include <string>

namespace energy_unit {

struct Transform {
    Eigen::Matrix4d matrix;      
    Eigen::Quaterniond rotation; 
    Eigen::Vector3d translation; 

    Transform() : matrix(Eigen::Matrix4d::Identity()),
                  rotation(Eigen::Quaterniond::Identity()),
                  translation(Eigen::Vector3d::Zero()) {}
};

class ArmTransform {
public:
    ArmTransform(const Transform& T_base_ee = Transform(), const Transform& T_ee_cam = Transform());
    void setBaseToEE(const Transform& T);
    void setEEToCamera(const Transform& T);
    Transform computeBaseToObject(const Transform& T_cam_obj) const;
    Transform computeEEToObject(const Transform& T_cam_obj) const;
    static Transform fromMatrix(const Eigen::Matrix4d& mat);
    static Transform fromEuler(double roll_deg, double pitch_deg, double yaw_deg, double tx, double ty, double tz);
private:
    Transform T_base_ee_;
    Transform T_ee_cam_;
};

}  // namespace energy_unit