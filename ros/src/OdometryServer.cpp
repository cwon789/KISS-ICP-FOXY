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

    // Subscriber
    // (변경) create_subscription<sensor_msgs::msg::PointCloud2>() 부분은 그대로지만,
    // RegisterFrame의 인자 타입을 pointer 값으로 받도록 바꿨으므로 문제 없음
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud_topic", rclcpp::SensorDataQoS(),
        std::bind(&OdometryServer::RegisterFrame, this, std::placeholders::_1));

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

// (변경) 여기서 인자 타입을 `const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg`에서
// `sensor_msgs::msg::PointCloud2::ConstSharedPtr msg` 형태로 수정
void OdometryServer::RegisterFrame(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);

    // KISS-ICP main pipeline
    const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(points, timestamps);

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
