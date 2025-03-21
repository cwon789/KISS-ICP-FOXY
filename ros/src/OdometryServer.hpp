#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <kiss_icp/pipeline/KissICP.hpp>
#include <sophus/se3.hpp>

// message_filters
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace kiss_icp_ros {

class OdometryServer : public rclcpp::Node {
public:
    explicit OdometryServer(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
    // (변경) 아래 인자에서 &를 제거해, 
    // callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) 형태로 수정
    void RegisterFrame(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void RegisterFrameWithInitGuess(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void SyncedCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
        const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg);
    void wheelOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void PublishOdometry(const Sophus::SE3d &kiss_pose, const std_msgs::msg::Header &header);
    void PublishClouds(const std::vector<Eigen::Vector3d> frame,
                       const std::vector<Eigen::Vector3d> keypoints,
                       const std_msgs::msg::Header &header);

    // --- message_filters ---
    // ApproximateTime Sync: PointCloud2 + WheelOdom
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloud_sub_;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odom_sub_;

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        nav_msgs::msg::Odometry> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // ROS parameters
    std::string base_frame_{"base_link"};
    std::string lidar_odom_frame_{"odom"};
    bool publish_odom_tf_{true};
    bool invert_odom_tf_{false};
    bool publish_debug_clouds_{false};
    double position_covariance_{0.1};
    double orientation_covariance_{0.1};

    // Subscriptions, publishers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kpoints_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;

    // Transform
    std::unique_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf2_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // KISS-ICP
    std::unique_ptr<kiss_icp::pipeline::KissICP> kiss_icp_;

    // Extrinsic
    Eigen::Isometry3d lidar_extrinsic_;

    // Wheel Callback
    Eigen::Isometry3d wheel_pose_ = Eigen::Isometry3d::Identity();
  
    // wheel odom 콜백에서 계산한 delta를 저장
    Eigen::Isometry3d last_delta_ = Eigen::Isometry3d::Identity();
    rclcpp::Time last_wheel_stamp_;
};

}  // namespace kiss_icp_ros
