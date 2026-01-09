// =================================================================
// 1. 定义 NOMINMAX
// =================================================================
#define NOMINMAX 
#define _USE_MATH_DEFINES 

#include <torch/torch.h>
#include <torch/script.h> 
#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h> 
#include <open3d/geometry/LineSet.h> 

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
// 【🔥 核心配置区 🔥】 (在这里修改方向)
// =================================================================

// 1. 如果头朝下脚朝上，把这个设为 true
const bool FIX_UPSIDE_DOWN = true;

// 2. 如果人是背对着你的 (屁股朝前)，把这个设为 true
const bool FIX_FRONT_BACK = false;

// =================================================================
// 2. 路径配置
// =================================================================
const std::string BASE_DIR = "D:/work/C++/my_cpp_project/";
const std::string MODEL_PATH = BASE_DIR + "smpl_female_30.pt"; // 或 male_gpu.pt
const std::string SCAN_PATH = BASE_DIR + "input.obj";
const std::string FACES_PATH = BASE_DIR + "smpl_faces.txt";

const std::string OUTPUT_DIR = BASE_DIR + "results/";
const std::string OUTPUT_OBJ = OUTPUT_DIR + "output_smpl_highres.glb";
const std::string OUTPUT_COMPARISON = OUTPUT_DIR + "result_comparison.glb";
const std::string OUTPUT_JOINTS_ONLY = OUTPUT_DIR + "joints_on_input.glb";
const std::string OUTPUT_LINES = OUTPUT_DIR + "measurement_lines.glb";

// 3. 参数配置
const float INIT_ARM_ANGLE = 1.0f;
const float INIT_Y_OFFSET = 0.05f;
const int NUM_BETAS = 30;

// ================= 辅助函数：几何计算 =================

double dist_v3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return (a - b).norm();
}

// 获取凸包排序后的索引
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

std::shared_ptr<geometry::LineSet> create_visual_ring(const std::vector<Eigen::Vector3d>& points, const Eigen::Vector3d& color, int axis_mode = 0) {
    auto lines = std::make_shared<geometry::LineSet>();
    if (points.size() < 3) return lines;
    auto indices = get_convex_hull_indices(points, axis_mode);
    for (size_t i = 0; i < indices.size(); ++i) {
        lines->points_.push_back(points[indices[i]]);
        lines->points_.push_back(points[indices[(i + 1) % indices.size()]]);
        lines->lines_.push_back(Eigen::Vector2i(2 * i, 2 * i + 1));
    }
    lines->PaintUniformColor(color);
    return lines;
}

// ================= 骨骼引导盒子切片 =================
std::vector<Eigen::Vector3d> get_slice_box(const std::shared_ptr<geometry::TriangleMesh>& mesh, double y, const Eigen::Vector3d& center_joint, double x_limit, int side_mode = 0) {
    std::vector<Eigen::Vector3d> slice;
    double tolerance = 0.02;
    for (const auto& v : mesh->vertices_) {
        if (std::abs(v.y() - y) < tolerance) {
            if (side_mode == 1 && v.x() < 0.005) continue;
            if (side_mode == -1 && v.x() > -0.005) continue;
            if (std::abs(v.x() - center_joint.x()) < x_limit) {
                slice.push_back(v);
            }
        }
    }
    return slice;
}

// ================= 核心：14项测量逻辑 =================
void measure_14_items(std::shared_ptr<geometry::TriangleMesh>& mesh, const torch::Tensor& joints_tensor) {
    torch::Tensor joints_cpu = joints_tensor.squeeze(0).cpu();
    auto joints_acc = joints_cpu.accessor<float, 2>();
    std::vector<Eigen::Vector3d> J(24);
    for (int i = 0; i < 24; ++i) J[i] = Eigen::Vector3d(joints_acc[i][0], joints_acc[i][1], joints_acc[i][2]);

    auto bbox = mesh->GetAxisAlignedBoundingBox();
    double min_y = bbox.GetMinBound().y();
    double height = bbox.GetMaxBound().y() - min_y;
    auto all_lines = std::make_shared<geometry::LineSet>();

    std::cout << "\n================ [ 14项专业身体测量报告 (v9.0) ] ================" << std::endl;
    std::cout << "身高           : " << height * 100.0 << " cm" << std::endl;

    // 1. 颈围 (Yellow)
    auto pts_neck = get_slice_box(mesh, J[12].y() + 0.08, J[12], 0.06);
    std::cout << "1)  颈围       : " << calculate_convex_hull_perimeter(pts_neck) * 100.0 << " cm" << std::endl;
    *all_lines += *create_visual_ring(pts_neck, Eigen::Vector3d(1.0, 1.0, 0.0));

    // 2. 胸围 (Green)
    auto pts_chest = get_slice_box(mesh, J[6].y(), J[6], 0.19); // 0.19 黄金宽度
    std::cout << "2)  胸围       : " << calculate_convex_hull_perimeter(pts_chest) * 100.0 << " cm" << std::endl;
    *all_lines += *create_visual_ring(pts_chest, Eigen::Vector3d(0.0, 1.0, 0.0));

    // 3. 腰围 (Magenta)
    Eigen::Vector3d waist_center = (J[3] + J[0]) * 0.5;
    double waist_girth = 1000.0;
    std::vector<Eigen::Vector3d> best_waist_pts;
    for (double y = J[0].y(); y < J[3].y(); y += 0.01) {
        auto pts = get_slice_box(mesh, y, waist_center, 0.22);
        double p = calculate_convex_hull_perimeter(pts);
        if (p > 0.5 && p < waist_girth) { waist_girth = p; best_waist_pts = pts; }
    }
    std::cout << "3)  腰围       : " << waist_girth * 100.0 << " cm" << std::endl;
    *all_lines += *create_visual_ring(best_waist_pts, Eigen::Vector3d(1.0, 0.0, 1.0));

    // 4. 臀围 (Orange)
    double hip_girth = 0.0;
    std::vector<Eigen::Vector3d> best_hip_pts;
    for (double y = J[0].y() - 0.15; y < J[0].y(); y += 0.01) {
        auto pts = get_slice_box(mesh, y, J[0], 0.23);
        double p = calculate_convex_hull_perimeter(pts);
        if (p > hip_girth) { hip_girth = p; best_hip_pts = pts; }
    }
    std::cout << "4)  臀围       : " << hip_girth * 100.0 << " cm" << std::endl;
    *all_lines += *create_visual_ring(best_hip_pts, Eigen::Vector3d(1.0, 0.5, 0.0));

    // 5. 大腿围 (Red)
    auto pts_thigh = get_slice_box(mesh, J[1].y() - 0.12, J[1], 0.13, 1);
    if (pts_thigh.size() < 50) pts_thigh = get_slice_box(mesh, J[1].y() - 0.08, J[1], 0.13, 1);
    std::cout << "5)  大腿围     : " << calculate_convex_hull_perimeter(pts_thigh) * 100.0 << " cm" << std::endl;
    *all_lines += *create_visual_ring(pts_thigh, Eigen::Vector3d(1.0, 0.0, 0.0));

    // 6. 肩宽
    double shoulder_dist = dist_v3(J[16], J[17]);
    std::cout << "6)  肩宽       : " << (shoulder_dist + 0.12) * 100.0 << " cm" << std::endl;
    {
        auto line = std::make_shared<geometry::LineSet>();
        line->points_ = { J[16], J[17] }; line->lines_ = { Eigen::Vector2i(0, 1) };
        line->PaintUniformColor(Eigen::Vector3d(0, 1, 1)); *all_lines += *line;
    }

    // 7. 袖笼围 (Blue)
    {
        double cut_x = J[16].x() + 0.02;
        double y_min = J[16].y() - 0.15; double y_max = J[16].y() + 0.10;
        double radius_limit = 0.13;
        std::vector<Eigen::Vector3d> pts_armhole;
        for (const auto& v : mesh->vertices_) {
            if (std::abs(v.x() - cut_x) < 0.02) {
                double dist_yz = std::sqrt(std::pow(v.y() - J[16].y(), 2) + std::pow(v.z() - J[16].z(), 2));
                if (dist_yz < radius_limit) pts_armhole.push_back(Eigen::Vector3d(v.y(), 0, v.z()));
            }
        }
        double armhole = calculate_convex_hull_perimeter(pts_armhole);
        std::vector<Eigen::Vector3d> visual_pts;
        for (auto& p : pts_armhole) visual_pts.push_back(Eigen::Vector3d(cut_x, p.x(), p.z()));
        std::cout << "7)  袖笼围     : " << armhole * 100.0 << " cm" << std::endl;
        *all_lines += *create_visual_ring(visual_pts, Eigen::Vector3d(0.0, 0.0, 1.0), 1);
    }

    // 8-14
    double arm_len = dist_v3(J[16], J[18]) + dist_v3(J[18], J[20]);
    std::cout << "8)  臂长       : " << (arm_len + 0.05) * 100.0 << " cm" << std::endl;
    double hand_len = height * 0.11;
    std::cout << "9)  中指长     : " << hand_len * 0.45 * 100.0 << " cm (AI)" << std::endl;
    std::cout << "10) 手长       : " << hand_len * 100.0 << " cm (AI)" << std::endl;
    std::cout << "11) 手宽       : " << hand_len * 0.48 * 100.0 << " cm (AI)" << std::endl;
    double torso_len = J[12].y() - J[0].y();
    std::cout << "12) 躯干垂直围 : " << (torso_len * 2.0 + 0.15) * 100.0 << " cm (估算)" << std::endl;
    double coat_len = J[12].y() - J[1].y();
    std::cout << "13) 衣长       : " << coat_len * 100.0 << " cm" << std::endl;
    double waist_h = (J[3].y() + J[0].y()) * 0.5;
    std::cout << "14) 裤长       : " << (waist_h - min_y) * 100.0 << " cm" << std::endl;
    std::cout << "==========================================================" << std::endl;

    io::WriteLineSet(OUTPUT_LINES, *all_lines);
    std::cout << "✅ 可视化线已保存: " << OUTPUT_LINES << std::endl;
}

// ================= 其他辅助函数 =================
torch::Tensor simple_chamfer_distance(torch::Tensor s, torch::Tensor t) {
    auto s_sq = s.squeeze(0);
    auto t_sq = t.squeeze(0);
    auto dist_mat = torch::cdist(s_sq, t_sq);
    return torch::mean(std::get<0>(torch::min(dist_mat, 1))) + torch::mean(std::get<0>(torch::min(dist_mat, 0)));
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
    torch::Tensor joints_cpu = joints.squeeze(0).cpu();
    auto acc = joints_cpu.accessor<float, 2>();
    for (int i = 0; i < 24; ++i) {
        auto sphere = geometry::TriangleMesh::CreateSphere(0.03, 10);
        sphere->Translate(Eigen::Vector3d(acc[i][0], acc[i][1], acc[i][2]));
        sphere->PaintUniformColor(color);
        *combined_mesh += *sphere;
    }
    return combined_mesh;
}

// ================= 主程序 =================
int main() {
    system("chcp 65001 > nul");
    try {
        if (!fs::exists(OUTPUT_DIR)) fs::create_directories(OUTPUT_DIR);

        // 1. CUDA 检测
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << ">>> 🚀 检测到 CUDA! 使用 GPU 加速..." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        // 2. 加载模型
        torch::jit::script::Module smpl_module = torch::jit::load(MODEL_PATH);
        smpl_module.to(device);

        // 3. 加载并预处理扫描数据
        std::cout << ">>> 正在加载扫描数据: " << SCAN_PATH << std::endl;
        auto target_mesh_ptr = io::CreateMeshFromFile(SCAN_PATH);
        if (!target_mesh_ptr || target_mesh_ptr->vertices_.empty()) throw std::runtime_error("读取失败");

        // --- 缩放 ---
        auto bbox = target_mesh_ptr->GetAxisAlignedBoundingBox();
        if ((bbox.GetMaxBound().y() - bbox.GetMinBound().y()) > 10.0) {
            std::cout << ">>> ⚠️ 缩放 x0.001" << std::endl;
            target_mesh_ptr->Scale(0.001, target_mesh_ptr->GetCenter());
        }

        // --- 核心预处理：姿态归一化 ---
        if (FIX_UPSIDE_DOWN) {
            std::cout << ">>> 🔄 修正上下颠倒..." << std::endl;
            Eigen::AngleAxisd rot_x(M_PI, Eigen::Vector3d::UnitX());
            target_mesh_ptr->Rotate(rot_x.toRotationMatrix(), target_mesh_ptr->GetCenter());
        }
        if (FIX_FRONT_BACK) {
            std::cout << ">>> 🔄 修正前后反向..." << std::endl;
            Eigen::AngleAxisd rot_y(M_PI, Eigen::Vector3d::UnitY());
            target_mesh_ptr->Rotate(rot_y.toRotationMatrix(), target_mesh_ptr->GetCenter());
        }

        // --- 居中 ---
        target_mesh_ptr->Translate(-target_mesh_ptr->GetAxisAlignedBoundingBox().GetCenter());
        target_mesh_ptr->PaintUniformColor(Eigen::Vector3d(0.6, 0.6, 0.6));

        // 4. 采样
        std::cout << ">>> 正在高密度采样 (50000点)..." << std::endl;
        auto pcd = target_mesh_ptr->SamplePointsUniformly(50000);
        auto vertices_eigen = pcd->points_;
        int num_verts = vertices_eigen.size();
        torch::Tensor target_verts = torch::zeros({ 1, num_verts, 3 }, torch::kCPU); // 先在CPU创建
        auto acc = target_verts.accessor<float, 3>();
        for (int i = 0; i < num_verts; ++i) {
            acc[0][i][0] = vertices_eigen[i].x();
            acc[0][i][1] = vertices_eigen[i].y();
            acc[0][i][2] = vertices_eigen[i].z();
        }
        target_verts = target_verts.to(device); // 传到 GPU

        // 5. 初始化 SMPL 变量
        auto betas = torch::zeros({ 1, NUM_BETAS }, device);
        auto pose = torch::zeros({ 1, 69 }, device);
        auto global_orient = torch::zeros({ 1, 3 }, device);
        auto transl = torch::tensor({ 0.0f, INIT_Y_OFFSET, 0.0f }, device).reshape({ 1, 3 });
        auto scale = torch::tensor({ 1.0 }, device);

        // 设置初始 A-Pose
        pose.index_put_({ 0, 47 }, -std::abs(INIT_ARM_ANGLE));
        pose.index_put_({ 0, 50 }, std::abs(INIT_ARM_ANGLE));

        betas.set_requires_grad(true); pose.set_requires_grad(true);
        global_orient.set_requires_grad(true); transl.set_requires_grad(true); scale.set_requires_grad(true);

        // 6. 优化
        std::vector<torch::Tensor> params = { transl, scale, betas, pose, global_orient };
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(0.02));

        std::cout << "\n>>> 开始优化..." << std::endl;
        for (int i = 0; i < 650; ++i) {
            optimizer.zero_grad();
            float w_pose = (i < 150) ? 10.0 : ((i < 450) ? 1.0 : 0.1);
            float w_beta = (i < 450 && i >= 150) ? 0.05 : 0.0;

            // 姿态锁
            if (i < 150) {
                torch::NoGradGuard no_grad;
                pose.data().index_put_({ 0, 47 }, -std::abs(INIT_ARM_ANGLE));
                pose.data().index_put_({ 0, 50 }, std::abs(INIT_ARM_ANGLE));
            }

            std::vector<torch::jit::IValue> inputs = { betas, pose, global_orient, transl, scale };
            auto out = smpl_module.forward(inputs).toTuple();
            auto smpl_verts = out->elements()[0].toTensor();

            auto loss = simple_chamfer_distance(smpl_verts, target_verts) +
                torch::mean(torch::pow(pose, 2)) * w_pose +
                torch::mean(torch::pow(betas, 2)) * w_beta;
            loss.backward();

            if (i < 150) { // 梯度遮罩
                pose.grad().index({ Slice(), Slice(3, torch::indexing::None) }).fill_(0.0);
                betas.grad().index({ Slice(), Slice(10, torch::indexing::None) }).fill_(0.0);
            }
            optimizer.step();
            if (i % 50 == 0) std::cout << "Iter " << i << " | Loss: " << loss.item<float>() << std::endl;
        }

        // 7. 保存与测量
        std::cout << "\n>>> 优化完成，正在保存..." << std::endl;
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = { betas, pose, global_orient, transl, scale };
        auto outputs = smpl_module.forward(inputs).toTuple();
        auto final_verts = outputs->elements()[0].toTensor().squeeze(0).cpu();
        auto final_joints = outputs->elements()[1].toTensor();

        auto smpl_mesh = std::make_shared<geometry::TriangleMesh>();
        auto v_acc = final_verts.accessor<float, 2>();
        for (int k = 0; k < final_verts.size(0); ++k) smpl_mesh->vertices_.push_back(Eigen::Vector3d(v_acc[k][0], v_acc[k][1], v_acc[k][2]));
        if (load_faces(FACES_PATH, smpl_mesh->triangles_)) {
            smpl_mesh->ComputeVertexNormals();
            io::WriteTriangleMesh(OUTPUT_OBJ, *smpl_mesh);
        }
        smpl_mesh->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));

        auto joints_vis = create_joints_visual(final_joints, Eigen::Vector3d(0.0, 1.0, 0.0));
        auto comp_mesh = std::make_shared<geometry::TriangleMesh>();
        *comp_mesh += *target_mesh_ptr; *comp_mesh += *smpl_mesh;
        io::WriteTriangleMesh(OUTPUT_COMPARISON, *comp_mesh);

        auto joints_check = std::make_shared<geometry::TriangleMesh>();
        *joints_check += *target_mesh_ptr; *joints_check += *joints_vis;
        io::WriteTriangleMesh(OUTPUT_JOINTS_ONLY, *joints_check);

        std::cout << "✅ 结果已保存至: " << OUTPUT_DIR << std::endl;
        measure_14_items(smpl_mesh, final_joints);

    }
    catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}