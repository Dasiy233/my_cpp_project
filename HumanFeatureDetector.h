#pragma once

// HumanFeatureDetector: encapsulates measurement logic + (optional) landmark detection.
// Units: all geometry is assumed to be in meters.

#include <torch/torch.h>
#include <open3d/Open3D.h>
#include <Eigen/Dense>

#include <array>
#include <memory>
#include <string>
#include <vector>

class HumanFeatureDetector {
public:
    struct VisParams {
        double vis_tol_y = 0.01;       // total thickness for visualization Y-slice (m)
        double vis_tol_x = 0.03;       // total thickness for visualization X-slice (m)
        double vis_sphere_r = 0.0008;  // point-sphere radius (m)
        double tube_r = 0.0010;        // default tube radius (m)
    };

    struct Report {
        // all in centimeters unless noted
        double height_cm = 0.0;
        double neck_cm = 0.0;
        double chest_cm = 0.0;
        double waist_cm = 0.0;
        double hip_cm = 0.0;
        double thigh_cm = 0.0;
        double shoulder_width_cm = 0.0;
        double armhole_cm = 0.0;
        double arm_len_cm = 0.0;
        double middle_finger_cm = 0.0;
        double hand_len_cm = 0.0;
        double hand_width_cm = 0.0;
        double torso_vertical_girth_cm = 0.0;
        double coat_len_cm = 0.0;
        double pants_len_cm = 0.0;
    };

    HumanFeatureDetector() = default;
    explicit HumanFeatureDetector(VisParams vis) : vis_(vis) {}

    // Main entry: performs 14-item measurement and writes a debug mesh with slice points/lines.
    // mesh is modified in-place (visual geometry appended) to match your current workflow.
    Report Measure14Items(
        const std::shared_ptr<open3d::geometry::TriangleMesh>& mesh,
        const torch::Tensor& joints_tensor,
        const std::string& output_merged_path);

private:
    struct Context {
        std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
        std::shared_ptr<open3d::geometry::TriangleMesh> visuals;
        VisParams vis;
        std::array<Eigen::Vector3d, 24> J;
        double min_y = 0.0;
        double height = 0.0;

        // carry across for pants_len cylinder
        double waist_y = 0.0;
        double waist_girth = 0.0;
        double max_waist_x = 0.0;
    };

    VisParams vis_;

    static std::array<Eigen::Vector3d, 24> TensorToJoints24(const torch::Tensor& joints_tensor);

    // 14 items (split into per-measurement methods)
    static double MeasureNeck(Context& ctx);
    static double MeasureChest(Context& ctx);
    static double MeasureWaist(Context& ctx);
    static double MeasureHip(Context& ctx);
    static double MeasureThigh(Context& ctx);
    static double MeasureShoulderWidth(Context& ctx);
    static double MeasureArmhole(Context& ctx);

    static void MeasureRest(Context& ctx, Report& rep);

    // ---- geometry helpers (kept private for now; later you can move to geo_utils.*) ----
    static std::vector<Eigen::Vector3d> GetSliceY(const open3d::geometry::TriangleMesh& mesh, double y_height, double tolerance);
    static std::vector<Eigen::Vector3d> GetSliceX(const open3d::geometry::TriangleMesh& mesh, double x_val, double tolerance);

    static std::vector<Eigen::Vector3d> GetTorsoSlice(
        const std::vector<Eigen::Vector3d>& full_slice_pts,
        double eps,
        int min_points,
        double thickness_ratio,
        bool debug);

    static std::vector<Eigen::Vector3d> GetConvexHullPointsXZ(const std::vector<Eigen::Vector3d>& points);
    static std::vector<Eigen::Vector3d> ResampleClosedRing(const std::vector<Eigen::Vector3d>& ring, int N);
    static double PerimeterClosed(const std::vector<Eigen::Vector3d>& ring);

    static std::shared_ptr<open3d::geometry::TriangleMesh> VisualizeSlicePoints(
        const std::vector<Eigen::Vector3d>& points,
        const Eigen::Vector3d& color,
        double radius);

    static std::shared_ptr<open3d::geometry::TriangleMesh> VisualizePathTube(
        const std::vector<Eigen::Vector3d>& path,
        const Eigen::Vector3d& color,
        double r);

    static double DistPointSegment(const Eigen::Vector3d& p, const Eigen::Vector3d& a, const Eigen::Vector3d& b);

};
