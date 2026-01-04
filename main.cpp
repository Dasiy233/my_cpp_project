// =================================================================
// 1. 定义 NOMINMAX
// =================================================================
#define NOMINMAX 
#define _USE_MATH_DEFINES 

#include <torch/torch.h>
#include <torch/script.h> 
#include <open3d/Open3D.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple> 
#include <cstdlib> 
#include <fstream> 
#include <filesystem> 
#include <numeric>
#include <algorithm>

using namespace torch::indexing;
using namespace open3d;
namespace fs = std::filesystem;

// =================================================================
// 【手动修正区】 
// =================================================================
// 0.0f   = 不旋转 (默认) -> 之前自动反了，现在不转试试
const float FORCE_ROTATION_ANGLE = 0.0f;

// =================================================================
// 2. 路径配置
// =================================================================
const std::string BASE_DIR = "D:/work/C++/my_cpp_project/";
const std::string MODEL_PATH = BASE_DIR + "smpl_female_30.pt";
const std::string SCAN_PATH = BASE_DIR + "input2.obj";
const std::string FACES_PATH = BASE_DIR + "smpl_faces.txt";

const std::string OUTPUT_DIR = BASE_DIR + "results/";
const std::string OUTPUT_OBJ = OUTPUT_DIR + "output_smpl_highres.glb";
const std::string OUTPUT_COMPARISON = OUTPUT_DIR + "result_comparison.glb";
const std::string OUTPUT_JOINTS_ONLY = OUTPUT_DIR + "joints_on_input.glb";

// 3. 参数配置
const float INIT_ARM_ANGLE = 1.0f;
const float INIT_Y_OFFSET = 0.05f;
const int NUM_BETAS = 30;

// ================= 辅助函数：几何计算 =================

double dist_v3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return (a - b).norm();
}

// 叉乘
double cross_product(const Eigen::Vector2d& O, const Eigen::Vector2d& A, const Eigen::Vector2d& B) {
    return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
}

// 凸包周长计算
double calculate_convex_hull_perimeter(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) return 0.0;

    std::vector<Eigen::Vector2d> P;
    for (const auto& p : points) P.push_back({ p.x(), p.z() });

    std::sort(P.begin(), P.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
        });

    std::vector<Eigen::Vector2d> H;
    // 下半壳
    for (const auto& p : P) {
        while (H.size() >= 2 && cross_product(H[H.size() - 2], H.back(), p) <= 0) H.pop_back();
        H.push_back(p);
    }
    // 上半壳
    size_t lower_size = H.size();
    for (int i = P.size() - 2; i >= 0; i--) {
        while (H.size() > lower_size && cross_product(H[H.size() - 2], H.back(), P[i]) <= 0) H.pop_back();
        H.push_back(P[i]);
    }
    H.pop_back();

    double perimeter = 0.0;
    for (size_t i = 0; i < H.size(); ++i) {
        perimeter += (H[(i + 1) % H.size()] - H[i]).norm();
    }
    return perimeter;
}

std::vector<Eigen::Vector3d> get_slice(const std::shared_ptr<geometry::TriangleMesh>& mesh, double y, int filter_mode = 0) {
    std::vector<Eigen::Vector3d> slice;
    double tolerance = 0.005;
    double torso_limit = 0.18;

    for (const auto& v : mesh->vertices_) {
        if (std::abs(v.y() - y) < tolerance) {
            if (filter_mode == 1) { // 躯干
                if (std::abs(v.x()) > torso_limit) continue;
            }
            else if (filter_mode == 2) { // 左侧
                if (v.x() < 0.02) continue;
            }
            else if (filter_mode == 3) { // 右侧
                if (v.x() > -0.02) continue;
            }
            slice.push_back(v);
        }
    }
    return slice;
}

// ================= 核心：14项测量逻辑 =================
void measure_14_items(std::shared_ptr<geometry::TriangleMesh>& mesh, const torch::Tensor& joints_tensor) {

    // 修复 E1776 和 维度错误
    torch::Tensor joints_cpu = joints_tensor.squeeze(0).cpu();
    auto joints_acc = joints_cpu.accessor<float, 2>();

    std::vector<Eigen::Vector3d> J(24);
    for (int i = 0; i < 24; ++i) J[i] = Eigen::Vector3d(joints_acc[i][0], joints_acc[i][1], joints_acc[i][2]);

    auto bbox = mesh->GetAxisAlignedBoundingBox();
    double min_y = bbox.GetMinBound().y();
    double max_y = bbox.GetMaxBound().y();
    double height = max_y - min_y;

    std::cout << "\n================ [ 14项专业身体测量报告 ] ================" << std::endl;
    std::cout << "身高           : " << height * 100.0 << " cm" << std::endl;

    // 1. 颈围
    auto pts_neck = get_slice(mesh, J[12].y() + 0.03, 1);
    std::cout << "1)  颈围       : " << calculate_convex_hull_perimeter(pts_neck) * 100.0 << " cm" << std::endl;

    // 2. 胸围
    auto pts_chest = get_slice(mesh, J[6].y(), 1);
    std::cout << "2)  胸围       : " << calculate_convex_hull_perimeter(pts_chest) * 100.0 << " cm" << std::endl;

    // 3. 腰围
    double waist_girth = 1000.0;
    for (double y = J[0].y(); y < J[3].y(); y += 0.01) {
        auto pts = get_slice(mesh, y, 1);
        double p = calculate_convex_hull_perimeter(pts);
        if (p > 0.5 && p < waist_girth) waist_girth = p;
    }
    std::cout << "3)  腰围       : " << waist_girth * 100.0 << " cm" << std::endl;

    // 4. 臀围
    double hip_girth = 0.0;
    for (double y = J[0].y() - 0.15; y < J[0].y(); y += 0.01) {
        auto pts = get_slice(mesh, y, 1);
        double p = calculate_convex_hull_perimeter(pts);
        if (p > hip_girth) hip_girth = p;
    }
    std::cout << "4)  臀围       : " << hip_girth * 100.0 << " cm" << std::endl;

    // 5. 大腿围
    auto pts_thigh = get_slice(mesh, J[1].y() - 0.03, 2);
    std::cout << "5)  大腿围     : " << calculate_convex_hull_perimeter(pts_thigh) * 100.0 << " cm" << std::endl;

    // 6. 肩宽
    double shoulder_dist = dist_v3(J[16], J[17]);
    std::cout << "6)  肩宽       : " << (shoulder_dist + 0.12) * 100.0 << " cm" << std::endl;

    // 7. 袖笼围
    double armhole = calculate_convex_hull_perimeter(get_slice(mesh, J[16].y(), 2));
    std::cout << "7)  袖笼围     : " << (armhole > 0 ? armhole : 0.40) * 100.0 << " cm (估算)" << std::endl;

    // 8. 臂长
    double arm_len = dist_v3(J[16], J[18]) + dist_v3(J[18], J[20]);
    std::cout << "8)  臂长       : " << (arm_len + 0.05) * 100.0 << " cm" << std::endl;

    // 9-11. 手部
    double hand_len = height * 0.11;
    std::cout << "9)  中指长     : " << hand_len * 0.45 * 100.0 << " cm (AI推算)" << std::endl;
    std::cout << "10) 手长       : " << hand_len * 100.0 << " cm (AI推算)" << std::endl;
    std::cout << "11) 手宽       : " << hand_len * 0.48 * 100.0 << " cm (AI推算)" << std::endl;

    // 12. 躯干垂直
    double torso_len = J[12].y() - J[0].y();
    std::cout << "12) 躯干垂直围 : " << (torso_len * 2.0 + 0.15) * 100.0 << " cm (估算)" << std::endl;

    // 13. 衣长
    double coat_len = J[12].y() - (J[1].y() - 0.1);
    std::cout << "13) 衣长       : " << coat_len * 100.0 << " cm" << std::endl;

    // 14. 裤长
    double waist_height = (J[3].y() + J[0].y()) * 0.5;
    double pants_len = waist_height - min_y;
    std::cout << "14) 裤长       : " << pants_len * 100.0 << " cm" << std::endl;

    std::cout << "==========================================================" << std::endl;
}

// ================= 其他辅助函数 =================
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
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << ">>> 🚀 检测到 CUDA! 使用 GPU 加速..." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        torch::jit::script::Module smpl_module = torch::jit::load(MODEL_PATH);
        smpl_module.to(device);

        std::cout << ">>> 正在加载扫描数据: " << SCAN_PATH << std::endl;
        auto target_mesh_ptr = io::CreateMeshFromFile(SCAN_PATH);
        if (target_mesh_ptr == nullptr || target_mesh_ptr->vertices_.empty()) throw std::runtime_error("读取扫描失败");

        auto bbox = target_mesh_ptr->GetAxisAlignedBoundingBox();
        if ((bbox.GetMaxBound().y() - bbox.GetMinBound().y()) > 10.0) {
            std::cout << ">>> ⚠️ 缩放 x0.001" << std::endl;
            target_mesh_ptr->Scale(0.001, target_mesh_ptr->GetCenter());
        }
        target_mesh_ptr->Translate(-target_mesh_ptr->GetAxisAlignedBoundingBox().GetCenter());
        target_mesh_ptr->PaintUniformColor(Eigen::Vector3d(0.6, 0.6, 0.6));

        // 强制旋转
        if (FORCE_ROTATION_ANGLE != 0.0f) {
            std::cout << ">>> [手动模式] 旋转 " << int(FORCE_ROTATION_ANGLE * 180 / M_PI) << " 度" << std::endl;
            Eigen::AngleAxisd rot_y(FORCE_ROTATION_ANGLE, Eigen::Vector3d::UnitY());
            target_mesh_ptr->Rotate(rot_y.toRotationMatrix(), target_mesh_ptr->GetCenter());
            target_mesh_ptr->Translate(-target_mesh_ptr->GetAxisAlignedBoundingBox().GetCenter());
        }

        // 3. 降采样
        auto pcd = target_mesh_ptr->SamplePointsUniformly(10000);
        auto vertices_eigen = pcd->points_;
        int num_verts = vertices_eigen.size();
        torch::Tensor target_verts_cpu = torch::zeros({ 1, num_verts, 3 }, torch::kCPU);
        auto acc = target_verts_cpu.accessor<float, 3>();
        for (int i = 0; i < num_verts; ++i) {
            acc[0][i][0] = static_cast<float>(vertices_eigen[i].x());
            acc[0][i][1] = static_cast<float>(vertices_eigen[i].y());
            acc[0][i][2] = static_cast<float>(vertices_eigen[i].z());
        }
        torch::Tensor target_verts = target_verts_cpu.to(device);

        // ================= 初始化变量 =================
        auto betas = torch::zeros({ 1, NUM_BETAS }, device); // 先不 grad
        auto pose = torch::zeros({ 1, 69 }, device);
        auto global_orient = torch::zeros({ 1, 3 }, device);
        auto transl = torch::tensor({ 0.0f, INIT_Y_OFFSET, 0.0f }, device).reshape({ 1, 3 });
        auto scale = torch::tensor({ 1.0 }, device);

        if (FORCE_ROTATION_ANGLE != 0.0f) {
            global_orient.index_put_({ 0, 1 }, FORCE_ROTATION_ANGLE);
        }

        // 初始 A-Pose
        pose.index_put_({ 0, 47 }, -std::abs(INIT_ARM_ANGLE));
        pose.index_put_({ 0, 50 }, std::abs(INIT_ARM_ANGLE));

        // 开启梯度
        betas.set_requires_grad(true);
        pose.set_requires_grad(true);
        global_orient.set_requires_grad(true);
        transl.set_requires_grad(true);
        scale.set_requires_grad(true);

        // ================= 优化循环 =================
        std::vector<torch::Tensor> params = { transl, scale, betas, pose, global_orient };
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(0.02));

        int total_iters = 650;
        std::cout << "\n>>> 开始优化..." << std::endl;

        for (int i = 0; i < total_iters; ++i) {
            optimizer.zero_grad();

            float w_pose_reg = 0.0; float w_beta_reg = 0.0;
            if (i < 150) {
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.02);
                w_pose_reg = 10.0; w_beta_reg = 0.0;

                // 【核心修复】删除了这里的报错代码
                // 仅依靠梯度锁即可，无需强制 reset

            }
            else if (i < 450) {
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.05);
                w_pose_reg = 1.0; w_beta_reg = 0.05;
            }
            else {
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.005);
                w_pose_reg = 0.1; w_beta_reg = 0.0;
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

            // 梯度锁
            bool lock_body_pose = (i < 150);
            bool lock_detail_betas = (i < 150);
            if (lock_body_pose) pose.grad().index({ Slice(), Slice(3, torch::indexing::None) }).fill_(0.0);
            if (lock_detail_betas) betas.grad().index({ Slice(), Slice(10, torch::indexing::None) }).fill_(0.0);

            optimizer.step();

            if (i % 50 == 0) std::cout << "Iter " << i << " | Dist: " << loss_dist.item<float>() << std::endl;
        }

        // ================= 保存与测量 =================
        std::cout << "\n>>> 优化完成，正在保存文件..." << std::endl;
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

        std::cout << "✅ 全部文件已保存至: " << OUTPUT_DIR << std::endl;

        measure_14_items(smpl_mesh_ptr, final_joints);

    }
    catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}