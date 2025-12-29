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

using namespace torch::indexing;
using namespace open3d;

// =================================================================
// 2. 路径配置 (绝对路径)
// =================================================================
const std::string MODEL_PATH = "D:/work/C++/my_cpp_project/smpl_female_30.pt";
const std::string SCAN_PATH = "D:/work/C++/my_cpp_project/input.obj";
const std::string FACES_PATH = "D:/work/C++/my_cpp_project/smpl_faces.txt";

// 【关键修改】全部改为 .glb 格式！
// .glb 格式完美支持颜色，且 Windows 自带 3D 查看器能直接打开
const std::string OUTPUT_OBJ = "D:/work/C++/my_cpp_project/output_smpl_highres.glb";
const std::string OUTPUT_COMPARISON = "D:/work/C++/my_cpp_project/result_comparison.glb";
const std::string OUTPUT_JOINTS_ONLY = "D:/work/C++/my_cpp_project/joints_on_input.glb";

// 3. 参数配置
const float INIT_ARM_ANGLE = 1.0f;
const float INIT_Y_OFFSET = 0.05f;
const int NUM_BETAS = 30;

// ================= 辅助函数 =================

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

// 创建关节球体可视化
std::shared_ptr<geometry::TriangleMesh> create_joints_visual(const torch::Tensor& joints, const Eigen::Vector3d& color) {
    auto combined_mesh = std::make_shared<geometry::TriangleMesh>();
    auto joints_cpu = joints.squeeze(0).cpu();
    auto acc = joints_cpu.accessor<float, 2>();

    // 只取前 24 个主要关节
    for (int i = 0; i < 24; ++i) {
        // 创建一个小球，半径 0.03 米
        auto sphere = geometry::TriangleMesh::CreateSphere(0.03, 10);
        sphere->Translate(Eigen::Vector3d(acc[i][0], acc[i][1], acc[i][2]));
        sphere->PaintUniformColor(color);
        *combined_mesh += *sphere; // 合并网格
    }
    return combined_mesh;
}

// ================= 主程序 =================
int main() {
    system("chcp 65001 > nul");

    try {
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << ">>> 🚀 检测到 CUDA! 使用 GPU 加速..." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        // 1. 加载模型
        torch::jit::script::Module smpl_module = torch::jit::load(MODEL_PATH);
        smpl_module.to(device);

        // 2. 加载原始扫描数据 (用于最终可视化)
        std::cout << ">>> 正在加载并预处理扫描数据..." << std::endl;
        auto target_mesh_ptr = io::CreateMeshFromFile(SCAN_PATH);
        if (target_mesh_ptr == nullptr || target_mesh_ptr->vertices_.empty()) throw std::runtime_error("读取扫描失败");

        // 预处理 (旋转 + 居中)
        Eigen::AngleAxisd rot_x(M_PI, Eigen::Vector3d::UnitX());
        target_mesh_ptr->Rotate(rot_x.toRotationMatrix(), target_mesh_ptr->GetCenter());
        Eigen::AngleAxisd rot_y(M_PI, Eigen::Vector3d::UnitY());
        target_mesh_ptr->Rotate(rot_y.toRotationMatrix(), target_mesh_ptr->GetCenter());
        target_mesh_ptr->Translate(-target_mesh_ptr->GetAxisAlignedBoundingBox().GetCenter());

        // 设置为灰色，准备用于可视化
        target_mesh_ptr->PaintUniformColor(Eigen::Vector3d(0.6, 0.6, 0.6)); // 灰色

        // 3. 降采样用于计算
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

        // ================= 初始化变量 =================
        auto betas = torch::zeros({ 1, NUM_BETAS }, torch::requires_grad().device(device));
        auto pose = torch::zeros({ 1, 69 }, torch::requires_grad().device(device));
        auto global_orient = torch::zeros({ 1, 3 }, torch::requires_grad().device(device));
        auto transl = torch::tensor({ 0.0f, INIT_Y_OFFSET, 0.0f }, device).reshape({ 1, 3 });
        transl.set_requires_grad(true);
        auto scale = torch::tensor({ 1.0 }, torch::requires_grad().device(device));

        // 初始 A-Pose
        {
            torch::NoGradGuard no_grad;
            pose.index_put_({ 0, 47 }, -std::abs(INIT_ARM_ANGLE));
            pose.index_put_({ 0, 50 }, std::abs(INIT_ARM_ANGLE));
        }

        // 自动朝向搜索
        std::cout << ">>> 搜索最佳水平朝向..." << std::endl;
        float best_loss = 1e9;
        float best_angle = 0.0f;
        std::vector<float> angles = { 0.0f, (float)M_PI / 2, (float)M_PI, -(float)M_PI / 2 };
        for (float ang : angles) {
            torch::NoGradGuard no_grad;
            auto temp_orient = global_orient.clone();
            temp_orient.index_put_({ 0, 1 }, ang);
            std::vector<torch::jit::IValue> in;
            in.push_back(betas); in.push_back(pose); in.push_back(temp_orient); in.push_back(transl); in.push_back(scale);
            auto v = smpl_module.forward(in).toTuple()->elements()[0].toTensor();
            float l = simple_chamfer_distance(v, target_verts).item<float>();
            if (l < best_loss) { best_loss = l; best_angle = ang; }
        }
        std::cout << "✅ 最佳角度: " << int(best_angle * 180 / M_PI) << " 度" << std::endl;
        { torch::NoGradGuard no_grad; global_orient.index_put_({ 0, 1 }, best_angle); }


        // ================= 优化循环 (1:1 复刻 Python 参数) =================
        std::vector<torch::Tensor> params = { transl, scale, betas, pose, global_orient };
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(0.02));

        int total_iters = 650;
        std::cout << "\n>>> 开始优化 (完全对齐 Python 逻辑)..." << std::endl;

        for (int i = 0; i < total_iters; ++i) {
            optimizer.zero_grad();

            float w_pose_reg = 0.0;
            float w_beta_reg = 0.0;
            std::string stage = "";
            bool lock_body_pose = false;
            bool lock_detail_betas = false;

            if (i < 150) {
                // Stage 1
                stage = "Stage 1: 刚性";
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.02);
                w_pose_reg = 10.0; w_beta_reg = 0.0;
                lock_body_pose = true; lock_detail_betas = true;
                { torch::NoGradGuard no_grad; pose.index_put_({ 0, 47 }, -std::abs(INIT_ARM_ANGLE)); pose.index_put_({ 0, 50 }, std::abs(INIT_ARM_ANGLE)); }

            }
            else if (i < 450) {
                // Stage 2
                stage = "Stage 2: 体型";
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.05);
                w_pose_reg = 1.0; w_beta_reg = 0.005;
                lock_body_pose = false; lock_detail_betas = false;

            }
            else {
                // Stage 3
                stage = "Stage 3: 微调";
                for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.005);
                w_pose_reg = 0.1; w_beta_reg = 0.0;
                lock_body_pose = false; lock_detail_betas = false;
            }

            // 前向传播
            std::vector<torch::jit::IValue> in;
            in.push_back(betas); in.push_back(pose); in.push_back(global_orient); in.push_back(transl); in.push_back(scale);
            auto out = smpl_module.forward(in).toTuple();
            auto smpl_verts = out->elements()[0].toTensor();

            // Loss 计算
            auto loss_dist = simple_chamfer_distance(smpl_verts, target_verts);
            auto loss_reg_pose = torch::mean(torch::pow(pose, 2)) * w_pose_reg;
            auto loss_reg_beta = torch::mean(torch::pow(betas, 2)) * w_beta_reg;
            auto total_loss = loss_dist + loss_reg_pose + loss_reg_beta;

            total_loss.backward();

            // 梯度锁
            if (lock_body_pose) pose.grad().index({ Slice(), Slice(3, torch::indexing::None) }).fill_(0.0);
            if (lock_detail_betas) betas.grad().index({ Slice(), Slice(10, torch::indexing::None) }).fill_(0.0);

            optimizer.step();

            if (i % 50 == 0) std::cout << "Iter " << i << " [" << stage << "] | Dist: " << loss_dist.item<float>() << std::endl;
        }

        // ================= 保存与可视化 =================
        std::cout << "\n>>> 优化完成，正在生成可视化结果..." << std::endl;
        torch::NoGradGuard no_grad;

        // 1. 获取最终的顶点和关节
        std::vector<torch::jit::IValue> in;
        in.push_back(betas); in.push_back(pose); in.push_back(global_orient); in.push_back(transl); in.push_back(scale);
        auto outputs = smpl_module.forward(in).toTuple();
        // elements()[0] 是 vertices, elements()[1] 是 joints
        auto final_verts = outputs->elements()[0].toTensor().squeeze(0).cpu();
        auto final_joints = outputs->elements()[1].toTensor(); // 保持在 GPU 上传给辅助函数

        // 2. 创建 SMPL Mesh (红色)
        auto smpl_mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        auto v_acc = final_verts.accessor<float, 2>();
        for (int k = 0; k < final_verts.size(0); ++k) smpl_mesh_ptr->vertices_.push_back(Eigen::Vector3d(v_acc[k][0], v_acc[k][1], v_acc[k][2]));
        if (load_faces(FACES_PATH, smpl_mesh_ptr->triangles_)) {
            smpl_mesh_ptr->ComputeVertexNormals();
            // 保存高精度模型
            // auto high_res_mesh = smpl_mesh_ptr->SubdivideLoop(1); // 可选细分
            io::WriteTriangleMesh(OUTPUT_OBJ, *smpl_mesh_ptr);
        }
        smpl_mesh_ptr->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0)); // 红色，半透明效果需要在Shader里做，这里先设为纯红

        // 3. 创建关节球体可视化 (绿色)
        auto joints_mesh_ptr = create_joints_visual(final_joints, Eigen::Vector3d(0.0, 1.0, 0.0));

        // 4. 生成对比文件 (红+灰)
        auto comparison_mesh = std::make_shared<geometry::TriangleMesh>();
        *comparison_mesh += *target_mesh_ptr; // 灰
        *comparison_mesh += *smpl_mesh_ptr;   // 红
        io::WriteTriangleMesh(OUTPUT_COMPARISON, *comparison_mesh);
        std::cout << "✅ 对比模型已保存: " << OUTPUT_COMPARISON << std::endl;

        // 5. 生成骨骼检查文件 (灰+绿)
        auto joints_check_mesh = std::make_shared<geometry::TriangleMesh>();
        *joints_check_mesh += *target_mesh_ptr; // 灰
        *joints_check_mesh += *joints_mesh_ptr; // 绿
        io::WriteTriangleMesh(OUTPUT_JOINTS_ONLY, *joints_check_mesh);
        std::cout << "✅ 骨骼检查模型已保存: " << OUTPUT_JOINTS_ONLY << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}