#include <Eigen/Core>
#include <memory>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

// KISS-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS 2 headers
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/exceptions.h>  // for tf2::TransformException

namespace {
// Foxy에서 tf2::TimePointZero 대신 rclcpp::Time(0)을 사용
// canTransform(...) 시 &err_msg를 인자로 직접 넘길 수 없어서 단순화
Sophus::SE3d LookupTransform(const std::string &target_frame,
                             const std::string &source_frame,
                             const std::unique_ptr<tf2_ros::Buffer> &tf2_buffer) {
    if (tf2_buffer->canTransform(target_frame, source_frame, rclcpp::Time(0))) {
        try {
            auto tf = tf2_buffer->lookupTransform(target_frame, source_frame, rclcpp::Time(0));
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "%s", ex.what());
        }
    }
    RCLCPP_WARN(rclcpp::get_logger("LookupTransform"),
                "Failed to find tf from %s to %s",
                source_frame.c_str(), target_frame.c_str());
    // 실패 시 identity 반환
    return Sophus::SE3d();
}
}  // namespace

namespace kiss_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

OdometryServer::OdometryServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("kiss_icp_node", options) {
    base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
    lidar_odom_frame_ = declare_parameter<std::string>("lidar_odom_frame", lidar_odom_frame_);
    publish_odom_tf_ = declare_parameter<bool>("publish_odom_tf", publish_odom_tf_);
    invert_odom_tf_ = declare_parameter<bool>("invert_odom_tf", invert_odom_tf_);
    publish_debug_clouds_ = declare_parameter<bool>("publish_debug_clouds", publish_debug_clouds_);
    position_covariance_ = declare_parameter<double>("position_covariance", 0.1);
    orientation_covariance_ = declare_parameter<double>("orientation_covariance", 0.1);

    // KISS-ICP configuration
    kiss_icp::pipeline::KISSConfig config;
    config.max_range = declare_parameter<double>("max_range", config.max_range);
    config.min_range = declare_parameter<double>("min_range", config.min_range);
    config.deskew = declare_parameter<bool>("deskew", config.deskew);
    config.voxel_size = declare_parameter<double>("voxel_size", config.max_range / 100.0);
    config.max_points_per_voxel =
        declare_parameter<int>("max_points_per_voxel", config.max_points_per_voxel);
    config.initial_threshold =
        declare_parameter<double>("initial_threshold", config.initial_threshold);
    config.min_motion_th = declare_parameter<double>("min_motion_th", config.min_motion_th);
    config.max_num_iterations =
        declare_parameter<int>("max_num_iterations", config.max_num_iterations);
    config.convergence_criterion =
        declare_parameter<double>("convergence_criterion", config.convergence_criterion);
    config.max_num_threads = declare_parameter<int>("max_num_threads", config.max_num_threads);

    if (config.max_range < config.min_range) {
        RCLCPP_WARN(get_logger(),
                    "[WARNING] max_range is smaller than min_range; setting min_range to 0.0");
        config.min_range = 0.0;
    }

    // Main KISS-ICP pipeline object
    kiss_icp_ = std::make_unique<kiss_icp::pipeline::KissICP>(config);
    wheel_pose_   = Eigen::Isometry3d::Identity();

    // [5] 파라미터로부터 LiDAR Extrinsic 설정
    double x     = declare_parameter<double>("lidar_extrinsic.x", 0.7237);
    double y_    = declare_parameter<double>("lidar_extrinsic.y", 0.0);
    double z     = declare_parameter<double>("lidar_extrinsic.z", 0.0);
    double roll  = declare_parameter<double>("lidar_extrinsic.roll", 0.0);
    double pitch = declare_parameter<double>("lidar_extrinsic.pitch", 0.0);
    double yaw   = declare_parameter<double>("lidar_extrinsic.yaw", 0.0);

    // 회전 변환: Rz * Ry * Rx (Z-Y-X 오일러 순)
    Eigen::AngleAxisd Rz(yaw,   Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rx(roll,  Eigen::Vector3d::UnitX());
    Eigen::Matrix3d rot = (Rz * Ry * Rx).matrix();

    lidar_extrinsic_ = Eigen::Isometry3d::Identity();
    lidar_extrinsic_.linear() = rot;
    lidar_extrinsic_.translation() = Eigen::Vector3d(x, y_, z);

    // Subscriber
    // (변경) create_subscription<sensor_msgs::msg::PointCloud2>() 부분은 그대로지만,
    // RegisterFrame의 인자 타입을 pointer 값으로 받도록 바꿨으므로 문제 없음
    // wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    //     "/bot_sensor/odometer/odometry", rclcpp::SensorDataQoS(),
    //     std::bind(&OdometryServer::wheelOdometryCallback, this, std::placeholders::_1)
    //   );

    // pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    //     "pointcloud_topic", rclcpp::SensorDataQoS(),
    //     std::bind(&OdometryServer::RegisterFrameWithInitGuess, this, std::placeholders::_1));

    cloud_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
        this, "pointcloud_topic", rclcpp::SensorDataQoS().get_rmw_qos_profile());

    odom_sub_  = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(
        this, "/bot_sensor/odometer/odometry", rclcpp::SensorDataQoS().get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10),
        *cloud_sub_, *odom_sub_);

    sync_->registerCallback(
            std::bind(&OdometryServer::SyncedCallback, this,
            std::placeholders::_1, std::placeholders::_2));

    // Publisher
    rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("kiss/odometry", qos);

    if (publish_debug_clouds_) {
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/frame", qos);
        kpoints_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/keypoints", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/local_map", qos);
    }

    // TF Broadcaster / Listener
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    // Foxy에서는 setUsingDedicatedThread(true) 불가
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);

    RCLCPP_INFO(this->get_logger(), "KISS-ICP ROS 2 odometry node initialized (Foxy-compatible)");
}


void OdometryServer::SyncedCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
    const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
{
    // 0) compute dt
    rclcpp::Time stamp(odom_msg->header.stamp, RCL_ROS_TIME);
    double dt = 0.0;
    if (last_wheel_stamp_.nanoseconds() != 0) {
        dt = (stamp - last_wheel_stamp_).seconds();
        if (dt < 0.0) {
            RCLCPP_WARN(get_logger(), "Negative dt => forcing dt=0");
            dt = 0.0;
        }
    }
    last_wheel_stamp_ = stamp;

    // 1) get (v, w)
    double v = odom_msg->twist.twist.linear.x;   // m/s
    double w = odom_msg->twist.twist.angular.z;  // rad/s

    // 2) unicycle exact integration => compute delta transform
    double yaw_before = std::atan2(last_delta_.rotation()(1, 0),
                                   last_delta_.rotation()(0, 0));

    // 여기서는 "delta"만 만드는 방식
    // 보통 last_delta_는 "이전 바퀴콜백" 대비의 증분이므로, 
    // 오히려 이건 "즉시 pointcloud에 적용"을 안 할 거라면
    // last_delta_를 매번 identity로 리셋 후, 이번 (v, w, dt)로 갱신하는 식
    // → 아래 예시는 "한 번의 wheel callback => 한 번의 delta" 개념
    last_delta_.setIdentity();

    if (std::fabs(w) < 1e-6) {
        // 직선
        double dx = v * dt * std::cos(yaw_before);
        double dy = v * dt * std::sin(yaw_before);
        Eigen::Isometry3d dT = Eigen::Isometry3d::Identity();
        dT.translate(Eigen::Vector3d(dx, dy, 0.0));
        last_delta_ = dT;
    } else {
        // arc
        double dtheta = w * dt;
        double yaw_after = yaw_before + dtheta;
        double R_ = v / w;
        double dx = R_ * (std::sin(yaw_after) - std::sin(yaw_before));
        double dy = -R_ * (std::cos(yaw_after) - std::cos(yaw_before));

        Eigen::Isometry3d dT = Eigen::Isometry3d::Identity();
        dT.translate(Eigen::Vector3d(dx, dy, 0.0));
        dT.rotate(Eigen::AngleAxisd(dtheta, Eigen::Vector3d::UnitZ()));
        last_delta_ = dT;
    }

    // ↑ 이렇게 하면 "직전 바퀴콜백" 이후 누적이 아닌, "이 콜백~다음 콜백 사이" 이동량
    //   단, 이 로직은 "콜백이 여러 번 올 때마다" last_delta_가 덮어씌워진다는 점 주의
    //   사실상 "가장 최신 바퀴 delta"만 저장

    // 2) PointCloud -> Eigen + Extrinsic
    auto points = PointCloud2ToEigen(cloud_msg);
    auto timestamps = GetTimestamps(cloud_msg);

    std::vector<Eigen::Vector3d> transformed_points;
    transformed_points.reserve(points.size());
    for (auto &pt : points) {
        transformed_points.push_back(lidar_extrinsic_ * pt);
    }

    // 3) wheel_pose_를 초기 추정으로 KISS-ICP 정합
    Sophus::SE3d guess_SE3(wheel_pose_.rotation(), wheel_pose_.translation());

    const auto &[frame, keypoints] = 
        kiss_icp_->RegisterFrameWithWheel(
            transformed_points,
            timestamps,
            guess_SE3
        );

    // 4) 최종 LiDAR pose
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // 5) Odometry publish
    PublishOdometry(kiss_pose, cloud_msg->header);

    // 6) Debug clouds
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, cloud_msg->header);
    }
}


void OdometryServer::wheelOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // 0) compute dt
    rclcpp::Time stamp(msg->header.stamp, RCL_ROS_TIME);
    double dt = 0.0;
    if (last_wheel_stamp_.nanoseconds() != 0) {
        dt = (stamp - last_wheel_stamp_).seconds();
        if (dt < 0.0) {
            RCLCPP_WARN(get_logger(), "Negative dt => forcing dt=0");
            dt = 0.0;
        }
    }
    last_wheel_stamp_ = stamp;

    // 1) get (v, w)
    double v = msg->twist.twist.linear.x;   // m/s
    double w = msg->twist.twist.angular.z;  // rad/s

    // 2) unicycle exact integration => compute delta transform
    double yaw_before = std::atan2(last_delta_.rotation()(1, 0),
                                   last_delta_.rotation()(0, 0));

    // 여기서는 "delta"만 만드는 방식
    // 보통 last_delta_는 "이전 바퀴콜백" 대비의 증분이므로, 
    // 오히려 이건 "즉시 pointcloud에 적용"을 안 할 거라면
    // last_delta_를 매번 identity로 리셋 후, 이번 (v, w, dt)로 갱신하는 식
    // → 아래 예시는 "한 번의 wheel callback => 한 번의 delta" 개념
    last_delta_.setIdentity();

    if (std::fabs(w) < 1e-6) {
        // 직선
        double dx = v * dt * std::cos(yaw_before);
        double dy = v * dt * std::sin(yaw_before);
        Eigen::Isometry3d dT = Eigen::Isometry3d::Identity();
        dT.translate(Eigen::Vector3d(dx, dy, 0.0));
        last_delta_ = dT;
    } else {
        // arc
        double dtheta = w * dt;
        double yaw_after = yaw_before + dtheta;
        double R_ = v / w;
        double dx = R_ * (std::sin(yaw_after) - std::sin(yaw_before));
        double dy = -R_ * (std::cos(yaw_after) - std::cos(yaw_before));

        Eigen::Isometry3d dT = Eigen::Isometry3d::Identity();
        dT.translate(Eigen::Vector3d(dx, dy, 0.0));
        dT.rotate(Eigen::AngleAxisd(dtheta, Eigen::Vector3d::UnitZ()));
        last_delta_ = dT;
    }

    // ↑ 이렇게 하면 "직전 바퀴콜백" 이후 누적이 아닌, "이 콜백~다음 콜백 사이" 이동량
    //   단, 이 로직은 "콜백이 여러 번 올 때마다" last_delta_가 덮어씌워진다는 점 주의
    //   사실상 "가장 최신 바퀴 delta"만 저장
}


// (변경) 여기서 인자 타입을 `const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg`에서
// `sensor_msgs::msg::PointCloud2::ConstSharedPtr msg` 형태로 수정
void OdometryServer::RegisterFrame(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);

    std::vector<Eigen::Vector3d> transformed_points;
    transformed_points.reserve(points.size());
    for (auto &pt : points) {
        transformed_points.push_back(lidar_extrinsic_ * pt);
    }

    // KISS-ICP main pipeline
    const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(transformed_points, timestamps);

    // LiDAR pose
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // Publish odom
    PublishOdometry(kiss_pose, msg->header);

    // Debug clouds
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, msg->header);
    }
}

void OdometryServer::RegisterFrameWithInitGuess(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);

    std::vector<Eigen::Vector3d> transformed_points;
    transformed_points.reserve(points.size());
    for (auto &pt : points) {
        transformed_points.push_back(lidar_extrinsic_ * pt);
    }

    Eigen::Isometry3d init_guess = wheel_pose_;
    Sophus::SE3d guess_SE3(init_guess.rotation(), init_guess.translation());

    // KISS-ICP main pipeline
    const auto &[frame, keypoints] =
        kiss_icp_->RegisterFrameWithWheel(
            transformed_points,
            timestamps,
            guess_SE3
        );
        
    // LiDAR pose
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // Publish odom
    PublishOdometry(kiss_pose, msg->header);

    // Debug clouds
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, msg->header);
    }
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &kiss_pose,
                                     const std_msgs::msg::Header &header) {
    const auto cloud_frame_id = header.frame_id;
    const bool egocentric_estimation = (base_frame_.empty() || base_frame_ == cloud_frame_id);
    const auto moving_frame = egocentric_estimation ? cloud_frame_id : base_frame_;

    // If we have a separate base_frame, transform the LiDAR-based pose into it
    const auto pose = [&]() -> Sophus::SE3d {
        if (egocentric_estimation) {
            return kiss_pose;
        } else {
            const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id, tf2_buffer_);
            return cloud2base * kiss_pose * cloud2base.inverse();
        }
    }();

    // TF broadcast
    if (publish_odom_tf_) {
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = header.stamp;

        if (invert_odom_tf_) {
            transform_msg.header.frame_id = moving_frame;
            transform_msg.child_frame_id = lidar_odom_frame_;
            transform_msg.transform = tf2::sophusToTransform(pose.inverse());
        } else {
            transform_msg.header.frame_id = lidar_odom_frame_;
            transform_msg.child_frame_id = moving_frame;
            transform_msg.transform = tf2::sophusToTransform(pose);
        }
        tf_broadcaster_->sendTransform(transform_msg);
    }

    // Odometry msg
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = lidar_odom_frame_;
    odom_msg.child_frame_id = moving_frame;
    odom_msg.pose.pose = tf2::sophusToPose(pose);

    // Covariance
    odom_msg.pose.covariance.fill(0.0);
    odom_msg.pose.covariance[0] = position_covariance_;
    odom_msg.pose.covariance[7] = position_covariance_;
    odom_msg.pose.covariance[14] = position_covariance_;
    odom_msg.pose.covariance[21] = orientation_covariance_;
    odom_msg.pose.covariance[28] = orientation_covariance_;
    odom_msg.pose.covariance[35] = orientation_covariance_;

    odom_publisher_->publish(std::move(odom_msg));
}

void OdometryServer::PublishClouds(const std::vector<Eigen::Vector3d> frame,
                                   const std::vector<Eigen::Vector3d> keypoints,
                                   const std_msgs::msg::Header &header) {
    const auto kiss_map = kiss_icp_->LocalMap();
    const auto kiss_pose = kiss_icp_->pose().inverse();

    frame_publisher_->publish(std::move(EigenToPointCloud2(frame, header)));
    kpoints_publisher_->publish(std::move(EigenToPointCloud2(keypoints, header)));

    auto local_map_header = header;
    local_map_header.frame_id = lidar_odom_frame_;
    map_publisher_->publish(std::move(EigenToPointCloud2(kiss_map, local_map_header)));
}

}  // namespace kiss_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kiss_icp_ros::OdometryServer)
