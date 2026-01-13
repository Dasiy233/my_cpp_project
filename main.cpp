// =================================================================
// 完整代码 v26.6-Mod (骨骼锚定 + 半径5.0cm 版)
// 1. [聚类] 骨骼锚定逻辑：只保留离骨骼最近的聚类
// 2. [参数] DBSCAN 半径调整为 5.0cm，防止胸部碎裂
// 3. [拟合] 用户指定的三阶段拟合逻辑
// =================================================================
#define NOMINMAX 
#define _USE_MATH_DEFINES 

#include <torch/torch.h>
#include <torch/script.h> 
#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h> 
#include <open3d/geometry/TriangleMesh.h> 

#include <iostream>
#include <vector>
#include <cmath>
#include <tuple> 
#include <cstdlib> 
#include <fstream> 
#include <filesystem> 
#include <numeric>
#include <algorithm>
#include <map> 

using namespace torch::indexing;
using namespace open3d;
namespace fs = std::filesystem;

// =================================================================
// 【🔥 核心配置区 🔥】
// =================================================================
const bool FIX_UPSIDE_DOWN = true;
const bool FIX_FRONT_BACK = true;
const float FORCE_ROTATION_ANGLE = 0.0f;

// 2. 路径配置
const std::string BASE_DIR = "D:/work/C++/my_cpp_project/";
const std::string MODEL_PATH = BASE_DIR + "smpl_female_30.pt";
const std::string SCAN_PATH = BASE_DIR + "input.obj";
const std::string FACES_PATH = BASE_DIR + "smpl_faces.txt";

const std::string OUTPUT_DIR = BASE_DIR + "results/";
const std::string OUTPUT_OBJ = OUTPUT_DIR + "output_smpl_highres.glb";
const std::string OUTPUT_COMPARISON = OUTPUT_DIR + "result_comparison.glb";
const std::string OUTPUT_JOINTS_ONLY = OUTPUT_DIR + "joints_on_input.glb";
const std::string OUTPUT_MERGED = OUTPUT_DIR + "model_with_lines.glb";

// 3. 参数配置
const float INIT_ARM_ANGLE = 1.0f;
const float INIT_Y_OFFSET = 0.05f;
const int NUM_BETAS = 30;

// ================= 辅助函数 =================

double dist_v3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return (a - b).norm();
}

std::vector<size_t> get_convex_hull_indices(const std::vector<Eigen::Vector3d>& points, int axis_mode = 0) {
    if (points.size() < 3) return {};
    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<Eigen::Vector2d> P;
    for (const auto& p : points) {
        if (axis_mode == 0) P.push_back({ p.x(), p.z() });
        else P.push_back({ p.y(), p.z() });
    }
    Eigen::Vector2d center(0, 0);
    for (auto& p : P) center += p;
    center /= (double)P.size();
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return std::atan2(P[i].y() - center.y(), P[i].x() - center.x()) <
            std::atan2(P[j].y() - center.y(), P[j].x() - center.x());
        });
    return indices;
}

double calculate_convex_hull_perimeter(const std::vector<Eigen::Vector3d>& points, int axis_mode = 0) {
    if (points.size() < 3) return 0.0;
    auto indices = get_convex_hull_indices(points, axis_mode);
    double perimeter = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        Eigen::Vector3d p1 = points[indices[i]];
        Eigen::Vector3d p2 = points[indices[(i + 1) % indices.size()]];
        if (axis_mode == 0) perimeter += std::sqrt(std::pow(p1.x() - p2.x(), 2) + std::pow(p1.z() - p2.z(), 2));
        else perimeter += std::sqrt(std::pow(p1.y() - p2.y(), 2) + std::pow(p1.z() - p2.z(), 2));
    }
    return perimeter;
}

std::vector<Eigen::Vector3d> smooth_path_loop(const std::vector<Eigen::Vector3d>& path, int iterations = 3) {
    if (path.size() < 3) return path;
    std::vector<Eigen::Vector3d> smoothed = path;
    for (int k = 0; k < iterations; ++k) {
        std::vector<Eigen::Vector3d> temp = smoothed;
        size_t n = smoothed.size();
        for (size_t i = 0; i < n; ++i) {
            size_t prev = (i == 0) ? n - 1 : i - 1;
            size_t next = (i == n - 1) ? 0 : i + 1;
            temp[i] = smoothed[prev] * 0.25 + smoothed[i] * 0.5 + smoothed[next] * 0.25;
        }
        smoothed = temp;
    }
    return smoothed;
}

std::shared_ptr<geometry::TriangleMesh> get_smooth_visual_ring(
    const std::vector<Eigen::Vector3d>& raw_points,
    const Eigen::Vector3d& color,
    double force_y_height,
    bool is_closed = true)
{
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (raw_points.size() < 3) return mesh;

    auto indices = get_convex_hull_indices(raw_points, 0);
    std::vector<Eigen::Vector3d> sorted_points;
    for (size_t idx : indices) {
        Eigen::Vector3d p = raw_points[idx];
        p.y() = force_y_height;
        sorted_points.push_back(p);
    }

    std::vector<Eigen::Vector3d> smoothed_points = smooth_path_loop(sorted_points, 5);

    double radius = 0.006;
    size_t n = smoothed_points.size();
    size_t count = is_closed ? n : n - 1;

    for (size_t i = 0; i < count; ++i) {
        Eigen::Vector3d p1 = smoothed_points[i];
        Eigen::Vector3d p2 = smoothed_points[(i + 1) % n];
        double len = (p1 - p2).norm();
        if (len < 1e-6) continue;

        auto cylinder = geometry::TriangleMesh::CreateCylinder(radius, len, 8);
        cylinder->PaintUniformColor(color);
        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d vec = (p2 - p1).normalized();
        Eigen::Vector3d axis = z_axis.cross(vec).normalized();
        double angle = std::acos(std::max(-1.0, std::min(1.0, z_axis.dot(vec))));
        if (std::abs(angle) > 1e-6) {
            Eigen::AngleAxisd rot(angle, axis);
            cylinder->Rotate(rot.toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
        }
        cylinder->Translate((p1 + p2) * 0.5);
        *mesh += *cylinder;

        auto sphere = geometry::TriangleMesh::CreateSphere(radius, 8);
        sphere->Translate(p1);
        sphere->PaintUniformColor(color);
        *mesh += *sphere;
    }
    return mesh;
}

std::vector<Eigen::Vector3d> get_slice_box(const std::shared_ptr<geometry::TriangleMesh>& mesh, double y) {
    std::vector<Eigen::Vector3d> slice;
    double tolerance = 0.02;
    for (const auto& v : mesh->vertices_) {
        if (std::abs(v.y() - y) < tolerance) {
            slice.push_back(v);
        }
    }
    return slice;
}

// =================================================================
// 【核心修改】骨骼锚定聚类 (半径增大到 5.0cm)
// =================================================================
std::vector<Eigen::Vector3d> KeepClusterClosestToJoint(
    const std::vector<Eigen::Vector3d>& input_points,
    const Eigen::Vector3d& joint_anchor,
    double cluster_eps = 0.05, // <--- 默认 5.0cm
    int min_points = 5)
{
    if (input_points.size() < min_points) return input_points;

    // 1. 单位自适应
    double max_val = 0.0;
    for (const auto& p : input_points) max_val = std::max(max_val, std::abs(p.y()));

    // 如果坐标值很大 (>100)，说明是毫米单位，改为 50.0mm
    if (max_val > 100.0) cluster_eps = 50.0;

    // 2. DBSCAN 聚类
    auto pcd = std::make_shared<geometry::PointCloud>(input_points);
    std::vector<int> labels = pcd->ClusterDBSCAN(cluster_eps, min_points, false);

    // 3. 寻找离 anchor (骨骼点) 最近的聚类
    int best_label = -1;
    double min_dist = DBL_MAX;

    std::map<int, std::vector<Eigen::Vector3d>> clusters;
    for (size_t i = 0; i < input_points.size(); ++i) {
        if (labels[i] != -1) {
            clusters[labels[i]].push_back(input_points[i]);
        }
    }

    if (clusters.empty()) return input_points;

    for (const auto& kv : clusters) {
        // 计算当前聚类的质心
        Eigen::Vector3d center(0, 0, 0);
        for (const auto& p : kv.second) center += p;
        center /= kv.second.size();

        // 计算质心到骨骼锚点的水平距离
        double dist = std::sqrt(std::pow(center.x() - joint_anchor.x(), 2) + std::pow(center.z() - joint_anchor.z(), 2));

        if (dist < min_dist) {
            min_dist = dist;
            best_label = kv.first;
        }
    }

    return clusters[best_label];
}


// 辅助函数
torch::Tensor simple_chamfer_distance(torch::Tensor s, torch::Tensor t) {
    auto s_sq = s.squeeze(0);
    auto t_sq = t.squeeze(0);
    auto dist_mat = torch::cdist(s_sq, t_sq);
    auto min_s2t = std::get<0>((torch::min)(dist_mat, 1));
    auto min_t2s = std::get<0>((torch::min)(dist_mat, 0));
    return torch::mean(min_s2t) + torch::mean(min_t2s);
}

bool load_faces(const std::string& path, std::vector<Eigen::Vector3i>& faces) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    int v1, v2, v3;
    while (file >> v1 >> v2 >> v3) faces.push_back(Eigen::Vector3i(v1, v2, v3));
    return true;
}

std::shared_ptr<geometry::TriangleMesh> create_joints_visual(const torch::Tensor& joints, const Eigen::Vector3d& color) {
    auto combined_mesh = std::make_shared<geometry::TriangleMesh>();
    auto joints_cpu = joints.squeeze(0).cpu();
    auto acc = joints_cpu.accessor<float, 2>();
    for (int i = 0; i < 24; ++i) {
        auto sphere = geometry::TriangleMesh::CreateSphere(0.03, 10);
        sphere->Translate(Eigen::Vector3d(acc[i][0], acc[i][1], acc[i][2]));
        sphere->PaintUniformColor(color);
        *combined_mesh += *sphere;
    }
    return combined_mesh;
}

// ================= 核心：14项测量逻辑 (骨骼锚定版) =================
void measure_14_items(std::shared_ptr<geometry::TriangleMesh>& mesh, const torch::Tensor& joints_tensor) {
    torch::Tensor joints_cpu = joints_tensor.squeeze(0).cpu();
    auto joints_acc = joints_cpu.accessor<float, 2>();
    std::vector<Eigen::Vector3d> J(24);
    for (int i = 0; i < 24; ++i) J[i] = Eigen::Vector3d(joints_acc[i][0], joints_acc[i][1], joints_acc[i][2]);

    auto bbox = mesh->GetAxisAlignedBoundingBox();
    double min_y = bbox.GetMinBound().y();
    double height = bbox.GetMaxBound().y() - min_y;
    auto all_visuals = std::make_shared<geometry::TriangleMesh>();

    std::cout << "\n================ [ 14项专业身体测量报告 (v26.6-Mod) ] ================" << std::endl;
    std::cout << "身高            : " << height * 100.0 << " cm" << std::endl;

    // 1. 颈围
    double neck_y = J[12].y() + 0.08;
    auto pts_neck = get_slice_box(mesh, neck_y);
    pts_neck = KeepClusterClosestToJoint(pts_neck, J[12]);
    std::cout << "1)  颈围        : " << calculate_convex_hull_perimeter(pts_neck) * 100.0 << " cm" << std::endl;
    *all_visuals += *get_smooth_visual_ring(pts_neck, Eigen::Vector3d(1.0, 1.0, 0.0), neck_y);

    // 2. 胸围 (骨骼锚定 + 半径5.0)
    double chest_y = J[6].y();
    auto pts_chest = get_slice_box(mesh, chest_y);
    pts_chest = KeepClusterClosestToJoint(pts_chest, J[6]);
    std::cout << "2)  胸围        : " << calculate_convex_hull_perimeter(pts_chest) * 100.0 << " cm" << std::endl;
    *all_visuals += *get_smooth_visual_ring(pts_chest, Eigen::Vector3d(0.0, 1.0, 0.0), chest_y);

    // 3. 腰围
    Eigen::Vector3d waist_center = (J[3] + J[0]) * 0.5;
    double waist_girth = 1000.0;
    std::vector<Eigen::Vector3d> best_waist_pts;
    double best_waist_y = 0;
    for (double y = J[0].y(); y < J[3].y(); y += 0.01) {
        auto pts = get_slice_box(mesh, y);
        pts = KeepClusterClosestToJoint(pts, waist_center);
        double p = calculate_convex_hull_perimeter(pts);
        if (p > 0.5 && p < waist_girth) { waist_girth = p; best_waist_pts = pts; best_waist_y = y; }
    }
    std::cout << "3)  腰围        : " << waist_girth * 100.0 << " cm" << std::endl;
    *all_visuals += *get_smooth_visual_ring(best_waist_pts, Eigen::Vector3d(1.0, 0.0, 1.0), best_waist_y);

    // 4. 臀围
    double hip_girth = 0.0;
    std::vector<Eigen::Vector3d> best_hip_pts;
    double best_hip_y = 0;
    for (double y = J[0].y() - 0.15; y < J[0].y(); y += 0.01) {
        auto pts = get_slice_box(mesh, y);
        pts = KeepClusterClosestToJoint(pts, J[0]);
        double p = calculate_convex_hull_perimeter(pts);
        if (p > hip_girth) { hip_girth = p; best_hip_pts = pts; best_hip_y = y; }
    }
    std::cout << "4)  臀围        : " << hip_girth * 100.0 << " cm" << std::endl;
    *all_visuals += *get_smooth_visual_ring(best_hip_pts, Eigen::Vector3d(1.0, 0.5, 0.0), best_hip_y);

    // 5. 大腿围
    double thigh_y = J[1].y() - 0.12;
    auto pts_thigh = get_slice_box(mesh, thigh_y);
    pts_thigh = KeepClusterClosestToJoint(pts_thigh, J[1]);
    if (pts_thigh.size() < 20) {
        thigh_y = J[1].y() - 0.08;
        pts_thigh = get_slice_box(mesh, thigh_y);
        pts_thigh = KeepClusterClosestToJoint(pts_thigh, J[1]);
    }
    std::cout << "5)  大腿围      : " << calculate_convex_hull_perimeter(pts_thigh) * 100.0 << " cm" << std::endl;
    *all_visuals += *get_smooth_visual_ring(pts_thigh, Eigen::Vector3d(1.0, 0.0, 0.0), thigh_y);

    // 6. 肩宽
    double shoulder_dist = dist_v3(J[16], J[17]);
    std::cout << "6)  肩宽        : " << (shoulder_dist + 0.12) * 100.0 << " cm" << std::endl;
    {
        auto cylinder = geometry::TriangleMesh::CreateCylinder(0.005, shoulder_dist, 8);
        cylinder->PaintUniformColor(Eigen::Vector3d(0, 1, 1));
        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d vec = (J[17] - J[16]).normalized();
        Eigen::Vector3d axis = z_axis.cross(vec).normalized();
        double angle = std::acos(z_axis.dot(vec));
        if (std::abs(angle) > 1e-6) {
            Eigen::AngleAxisd rot(angle, axis);
            cylinder->Rotate(rot.toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
        }
        cylinder->Translate((J[16] + J[17]) * 0.5);
        *all_visuals += *cylinder;
    }

    // 7. 袖笼围
    {
        double cut_x = J[16].x() + 0.02;
        double radius_limit = 0.13;
        std::vector<Eigen::Vector3d> pts_armhole;
        for (const auto& v : mesh->vertices_) {
            if (std::abs(v.x() - cut_x) < 0.02) {
                double dist_yz = std::sqrt(std::pow(v.y() - J[16].y(), 2) + std::pow(v.z() - J[16].z(), 2));
                if (dist_yz < radius_limit) pts_armhole.push_back(Eigen::Vector3d(v.y(), 0, v.z()));
            }
        }
        pts_armhole = KeepClusterClosestToJoint(pts_armhole, J[16]);
        double armhole = calculate_convex_hull_perimeter(pts_armhole);
        std::cout << "7)  袖笼围      : " << armhole * 100.0 << " cm" << std::endl;

        std::vector<Eigen::Vector3d> visual_pts;
        for (auto& p : pts_armhole) visual_pts.push_back(Eigen::Vector3d(cut_x, p.x(), p.z()));
        auto indices = get_convex_hull_indices(visual_pts, 1);
        std::vector<Eigen::Vector3d> sorted;
        for (auto idx : indices) sorted.push_back(visual_pts[idx]);
        std::vector<Eigen::Vector3d> smoothed = smooth_path_loop(sorted, 3);
        double r = 0.004;
        for (size_t i = 0; i < smoothed.size(); ++i) {
            Eigen::Vector3d p1 = smoothed[i];
            Eigen::Vector3d p2 = smoothed[(i + 1) % smoothed.size()];
            double len = (p1 - p2).norm();
            if (len < 1e-6) continue;
            auto cyl = geometry::TriangleMesh::CreateCylinder(r, len, 8);
            cyl->PaintUniformColor(Eigen::Vector3d(0, 0, 1));
            Eigen::Vector3d z(0, 0, 1);
            Eigen::Vector3d v = (p2 - p1).normalized();
            Eigen::Vector3d ax = z.cross(v).normalized();
            double ang = std::acos(std::max(-1.0, std::min(1.0, z.dot(v))));
            if (std::abs(ang) > 1e-6) cyl->Rotate(Eigen::AngleAxisd(ang, ax).toRotationMatrix(), { 0,0,0 });
            cyl->Translate((p1 + p2) * 0.5);
            *all_visuals += *cyl;
        }
    }

    // 8-14
    double arm_len = dist_v3(J[16], J[18]) + dist_v3(J[18], J[20]);
    std::cout << "8)  臂长        : " << (arm_len + 0.05) * 100.0 << " cm" << std::endl;
    double hand_len = height * 0.11;
    std::cout << "9)  中指长      : " << hand_len * 0.45 * 100.0 << " cm (AI)" << std::endl;
    std::cout << "10) 手长        : " << hand_len * 100.0 << " cm (AI)" << std::endl;
    std::cout << "11) 手宽        : " << hand_len * 0.48 * 100.0 << " cm (AI)" << std::endl;
    double torso_len = J[12].y() - J[0].y();
    std::cout << "12) 躯干垂直围 : " << (torso_len * 2.0 + 0.15) * 100.0 << " cm (估算)" << std::endl;
    double coat_len = J[12].y() - J[1].y();
    std::cout << "13) 衣长        : " << coat_len * 100.0 << " cm" << std::endl;
    double waist_h = (J[3].y() + J[0].y()) * 0.5;
    std::cout << "14) 裤长        : " << (waist_h - min_y) * 100.0 << " cm" << std::endl;
    std::cout << "==========================================================" << std::endl;

    *mesh += *all_visuals;
    io::WriteTriangleMesh(OUTPUT_MERGED, *mesh);
    std::cout << "✅ 最终合体模型(含测量线)已保存: " << OUTPUT_MERGED << std::endl;
}

// ================= 主程序 =================
int main() {
    system("chcp 65001 > nul");

    try {
        if (!fs::exists(OUTPUT_DIR)) fs::create_directories(OUTPUT_DIR);

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << ">>> 🚀 检测到 CUDA! 使用 GPU 加速..." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        // 1. 加载模型
        torch::jit::script::Module smpl_module = torch::jit::load(MODEL_PATH);
        smpl_module.to(device);

        // 2. 加载扫描数据
        std::cout << ">>> 正在加载扫描数据: " << SCAN_PATH << std::endl;
        auto target_mesh_ptr = io::CreateMeshFromFile(SCAN_PATH);
        if (target_mesh_ptr == nullptr || target_mesh_ptr->vertices_.empty()) throw std::runtime_error("读取扫描失败");

        // --- 预处理 ---
        auto bbox = target_mesh_ptr->GetAxisAlignedBoundingBox();
        double height = bbox.GetMaxBound().y() - bbox.GetMinBound().y();
        if (height > 10.0) {
            std::cout << ">>> ⚠️ 检测到毫米单位，缩放 x0.001" << std::endl;
            target_mesh_ptr->Scale(0.001, target_mesh_ptr->GetCenter());
        }

        if (FIX_UPSIDE_DOWN) {
            std::cout << ">>> 🔄 修正上下颠倒 (Rotate X 180)..." << std::endl;
            Eigen::AngleAxisd rot_x(M_PI, Eigen::Vector3d::UnitX());
            target_mesh_ptr->Rotate(rot_x.toRotationMatrix(), target_mesh_ptr->GetCenter());
        }
        if (FIX_FRONT_BACK) {
            std::cout << ">>> 🔄 修正前后反向 (Rotate Y 180)..." << std::endl;
            Eigen::AngleAxisd rot_y(M_PI, Eigen::Vector3d::UnitY());
            target_mesh_ptr->Rotate(rot_y.toRotationMatrix(), target_mesh_ptr->GetCenter());
        }

        target_mesh_ptr->Translate(-target_mesh_ptr->GetAxisAlignedBoundingBox().GetCenter());
        target_mesh_ptr->PaintUniformColor(Eigen::Vector3d(0.6, 0.6, 0.6));

        auto pcd = target_mesh_ptr->SamplePointsUniformly(10000);
        auto vertices_eigen = pcd->points_;
        int num_verts = vertices_eigen.size();

        torch::Tensor target_verts_cpu = torch::zeros({ 1, num_verts, 3 }, torch::kCPU);
        auto accessor = target_verts_cpu.accessor<float, 3>();
        for (int i = 0; i < num_verts; ++i) {
            accessor[0][i][0] = static_cast<float>(vertices_eigen[i].x());
            accessor[0][i][1] = static_cast<float>(vertices_eigen[i].y());
            accessor[0][i][2] = static_cast<float>(vertices_eigen[i].z());
        }
        torch::Tensor target_verts = target_verts_cpu.to(device);

        auto betas = torch::zeros({ 1, NUM_BETAS }, torch::requires_grad().device(device));
        auto pose = torch::zeros({ 1, 69 }, torch::requires_grad().device(device));
        auto global_orient = torch::zeros({ 1, 3 }, torch::requires_grad().device(device));
        auto transl = torch::tensor({ 0.0f, INIT_Y_OFFSET, 0.0f }, device).reshape({ 1, 3 });
        transl.set_requires_grad(true);
        auto scale = torch::tensor({ 1.0 }, torch::requires_grad().device(device));

        {
            torch::NoGradGuard no_grad;
            pose.index_put_({ 0, 47 }, -std::abs(INIT_ARM_ANGLE));
            pose.index_put_({ 0, 50 }, std::abs(INIT_ARM_ANGLE));
        }

        std::cout << ">>> [手动模式] 强制设定初始朝向: " << int(FORCE_ROTATION_ANGLE * 180 / M_PI) << " 度" << std::endl;
        {
            torch::NoGradGuard no_grad;
            global_orient.index_put_({ 0, 1 }, FORCE_ROTATION_ANGLE);
        }

        std::vector<torch::Tensor> params = { transl, scale, betas, pose, global_orient };
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(0.02));

        int total_iters = 650;
        std::cout << "\n>>> 开始优化 (结果将存入 " << OUTPUT_DIR << ")..." << std::endl;

        for (int i = 0; i < total_iters; ++i) {
            optimizer.zero_grad();

            float w_pose_reg = 0.0;
            float w_beta_reg = 0.0;
            std::string stage = "";
            bool lock_body_pose = false;
            bool lock_detail_betas = false;

            if (i < 150) {
                stage = "Stage 1: 刚性";
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.02);
                w_pose_reg = 10.0; w_beta_reg = 0.0;
                lock_body_pose = true; lock_detail_betas = true;
                { torch::NoGradGuard no_grad; pose.index_put_({ 0, 47 }, -std::abs(INIT_ARM_ANGLE)); pose.index_put_({ 0, 50 }, std::abs(INIT_ARM_ANGLE)); }
            }
            else if (i < 450) {
                stage = "Stage 2: 体型";
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.05);
                w_pose_reg = 1.0;
                w_beta_reg = 0.05;
                lock_body_pose = false; lock_detail_betas = false;
            }
            else {
                stage = "Stage 3: 微调";
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.005);
                w_pose_reg = 0.1; w_beta_reg = 0.0;
                lock_body_pose = false; lock_detail_betas = false;
            }

            std::vector<torch::jit::IValue> in;
            in.push_back(betas); in.push_back(pose); in.push_back(global_orient); in.push_back(transl); in.push_back(scale);
            auto out = smpl_module.forward(in).toTuple();
            auto smpl_verts = out->elements()[0].toTensor();

            auto loss_dist = simple_chamfer_distance(smpl_verts, target_verts);
            auto loss_reg_pose = torch::mean(torch::pow(pose, 2)) * w_pose_reg;
            auto loss_reg_beta = torch::mean(torch::pow(betas, 2)) * w_beta_reg;
            auto total_loss = loss_dist + loss_reg_pose + loss_reg_beta;

            total_loss.backward();

            if (lock_body_pose) pose.grad().index({ Slice(), Slice(3, torch::indexing::None) }).fill_(0.0);
            if (lock_detail_betas) betas.grad().index({ Slice(), Slice(10, torch::indexing::None) }).fill_(0.0);

            optimizer.step();

            if (i % 50 == 0) std::cout << "Iter " << i << " [" << stage << "] | Dist: " << loss_dist.item<float>() << std::endl;
        }

        std::cout << "\n>>> 优化完成，正在保存..." << std::endl;
        torch::NoGradGuard no_grad;

        std::vector<torch::jit::IValue> in;
        in.push_back(betas); in.push_back(pose); in.push_back(global_orient); in.push_back(transl); in.push_back(scale);
        auto outputs = smpl_module.forward(in).toTuple();
        auto final_verts = outputs->elements()[0].toTensor().squeeze(0).cpu();
        auto final_joints = outputs->elements()[1].toTensor();

        auto smpl_mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        auto v_acc = final_verts.accessor<float, 2>();
        for (int k = 0; k < final_verts.size(0); ++k) smpl_mesh_ptr->vertices_.push_back(Eigen::Vector3d(v_acc[k][0], v_acc[k][1], v_acc[k][2]));
        if (load_faces(FACES_PATH, smpl_mesh_ptr->triangles_)) {
            smpl_mesh_ptr->ComputeVertexNormals();
            io::WriteTriangleMesh(OUTPUT_OBJ, *smpl_mesh_ptr);
        }
        smpl_mesh_ptr->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));

        auto joints_mesh_ptr = create_joints_visual(final_joints, Eigen::Vector3d(0.0, 1.0, 0.0));

        auto comparison_mesh = std::make_shared<geometry::TriangleMesh>();
        *comparison_mesh += *target_mesh_ptr;
        *comparison_mesh += *smpl_mesh_ptr;
        io::WriteTriangleMesh(OUTPUT_COMPARISON, *comparison_mesh);

        auto joints_check_mesh = std::make_shared<geometry::TriangleMesh>();
        *joints_check_mesh += *target_mesh_ptr;
        *joints_check_mesh += *joints_mesh_ptr;
        io::WriteTriangleMesh(OUTPUT_JOINTS_ONLY, *joints_check_mesh);

        measure_14_items(smpl_mesh_ptr, final_joints);

    }
    catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}