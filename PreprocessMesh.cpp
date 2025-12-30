// =================================================================
// 1. 宏定义与头文件
// =================================================================
#define NOMINMAX 
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <memory>

#include <Eigen/Dense>
#include <open3d/Open3D.h>

// 强制移除 Windows 的 min/max 宏
#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif

using namespace open3d;

// =========================================================
// 2. 辅助函数
// =========================================================

double PointToSegmentDistance(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d ap = p - a;
    double l2 = ab.squaredNorm();
    if (l2 == 0.0) return ap.norm();
    double t = std::max(0.0, std::min(1.0, ap.dot(ab) / l2));
    Eigen::Vector2d projection = a + t * ab;
    return (p - projection).norm();
}

std::pair<double, double> FitLine2D(const std::vector<Eigen::Vector2d>& points) {
    if (points.size() < 2) return { 0.0, 0.0 };
    int n = static_cast<int>(points.size());
    Eigen::MatrixXd A(n, 2);
    Eigen::VectorXd B(n);
    for (int i = 0; i < n; ++i) {
        A(i, 0) = points[i].x();
        A(i, 1) = 1.0;
        B(i) = points[i].y();
    }
    Eigen::Vector2d coeffs = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    return { coeffs[0], coeffs[1] };
}

double CalculateConvexHullPerimeter(const std::vector<Eigen::Vector3d>& slice_points) {
    if (slice_points.size() < 3) return 0.0;
    std::vector<Eigen::Vector2d> points_2d;
    points_2d.reserve(slice_points.size());
    for (const auto& pt3d : slice_points) points_2d.emplace_back(pt3d.x(), pt3d.z());

    auto cross_product = [](const Eigen::Vector2d& O, const Eigen::Vector2d& A, const Eigen::Vector2d& B) {
        return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
        };

    std::sort(points_2d.begin(), points_2d.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
        });

    std::vector<Eigen::Vector2d> hull;
    for (const auto& p : points_2d) {
        while (hull.size() >= 2 && cross_product(hull[hull.size() - 2], hull.back(), p) <= 0) hull.pop_back();
        hull.push_back(p);
    }
    for (int i = static_cast<int>(points_2d.size()) - 2, t = static_cast<int>(hull.size()) + 1; i >= 0; i--) {
        while (hull.size() >= t && cross_product(hull[hull.size() - 2], hull.back(), points_2d[i]) <= 0) hull.pop_back();
        hull.push_back(points_2d[i]);
    }
    hull.pop_back();

    double perimeter = 0.0;
    for (size_t i = 0; i < hull.size(); ++i) perimeter += (hull[(i + 1) % hull.size()] - hull[i]).norm();
    return perimeter;
}

std::vector<Eigen::Vector3d> GetSlice(double y_height, double tolerance, const geometry::TriangleMesh& mesh_to_slice) {
    std::vector<Eigen::Vector3d> slice_points;
    const double min_y = y_height - tolerance / 2.0;
    const double max_y = y_height + tolerance / 2.0;
    for (const auto& vertex : mesh_to_slice.vertices_) {
        if (vertex.y() >= min_y && vertex.y() <= max_y) slice_points.push_back(vertex);
    }
    return slice_points;
}

// =========================================================
// 3. 核心处理逻辑
// =========================================================

// 【新增功能】最后检查是否倒立，如果倒立则翻转
void CheckAndFlipHuman(std::shared_ptr<geometry::TriangleMesh>& mesh) {
    auto bbox = mesh->GetAxisAlignedBoundingBox();
    double min_y = bbox.GetMinBound().y();
    double max_y = bbox.GetMaxBound().y();
    double height = max_y - min_y;

    // 逻辑：将身体分成 10 段，找出最宽的那一段（bounding box width）
    double max_width = -1.0;
    double y_of_max_width = min_y;

    int steps = 10;
    for (int i = 0; i < steps; ++i) {
        double y_center = min_y + (height / steps) * (i + 0.5);
        auto slice = GetSlice(y_center, height / steps, *mesh);
        if (slice.empty()) continue;

        // 计算这一层的 X 轴宽度
        double min_x = 1e9, max_x = -1e9;
        for (const auto& v : slice) {
            min_x = std::min(min_x, v.x());
            max_x = std::max(max_x, v.x());
        }
        double width = max_x - min_x;
        if (width > max_width) {
            max_width = width;
            y_of_max_width = y_center;
        }
    }

    // 判断：人体的最宽处（肩膀/手臂）通常在上半身 (> 0.5 高度)
    // 如果最宽处跑到了下半身 (< 0.4 高度)，说明可能倒立了
    double relative_pos = (y_of_max_width - min_y) / height;
    std::cout << ">>> 人体形态检查: 最宽处位于高度比例 " << relative_pos * 100.0 << "% 处。" << std::endl;

    if (relative_pos < 0.4) {
        std::cout << ">>> ⚠️ 检测到人体倒立 (最宽处在下半身)！正在自动翻转..." << std::endl;

        // 绕 X 轴旋转 180 度
        Eigen::AngleAxisd rot_x(M_PI, Eigen::Vector3d::UnitX());
        mesh->Rotate(rot_x.toRotationMatrix(), mesh->GetCenter());

        // 重新落地
        auto new_bbox = mesh->GetAxisAlignedBoundingBox();
        mesh->Translate(Eigen::Vector3d(0, -new_bbox.GetMinBound().y(), 0));
        std::cout << "✅ 已修正倒立问题。" << std::endl;
    }
    else {
        std::cout << "✅ 人体方向正确。" << std::endl;
    }
}

void FixOrientationAndNormalize(std::shared_ptr<geometry::TriangleMesh>& mesh) {
    std::cout << ">>> 开始处理模型 (去地面 + 姿态校正)..." << std::endl;
    if (mesh->vertices_.empty()) return;
    Eigen::Vector3d initial_centroid = mesh->GetCenter();

    // 1. 第一次 PCA
    Eigen::Matrix3d covariance1 = Eigen::Matrix3d::Zero();
    for (const auto& v : mesh->vertices_) covariance1 += (v - initial_centroid) * (v - initial_centroid).transpose();
    if (mesh->vertices_.size() > 1) covariance1 /= static_cast<double>(mesh->vertices_.size() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver1(covariance1);
    Eigen::Matrix3d pca1_principal_axes = eigen_solver1.eigenvectors();
    Eigen::Vector3d pca1_height_axis = pca1_principal_axes.col(2); // Z轴
    Eigen::Vector3d pca1_width_axis = pca1_principal_axes.col(1);

    // 2. 地面检测
    auto analyze_end = [&](const geometry::TriangleMesh& m, const Eigen::Vector3d& h_axis, bool is_max_end) -> std::vector<size_t> {
        const auto& vertices = m.vertices_;
        if (vertices.empty()) return {};
        double min_h = 1e9, max_h = -1e9;
        for (const auto& v : vertices) { double h = v.dot(h_axis); min_h = std::min(min_h, h); max_h = std::max(max_h, h); }

        std::vector<size_t> seed_indices;
        std::vector<Eigen::Vector3d> seed_points;
        double slice_thickness = 15.0;
        if (is_max_end) {
            for (size_t i = 0; i < vertices.size(); ++i) if (vertices[i].dot(h_axis) > max_h - slice_thickness) seed_indices.push_back(i);
        }
        else {
            for (size_t i = 0; i < vertices.size(); ++i) if (vertices[i].dot(h_axis) < min_h + slice_thickness) seed_indices.push_back(i);
        }

        if (seed_indices.size() < 200) return {};
        for (size_t idx : seed_indices) seed_points.push_back(vertices[idx]);

        double h_sum = 0.0; for (const auto& pt : seed_points) h_sum += pt.dot(h_axis);
        double h_mean = h_sum / static_cast<double>(seed_points.size());
        double h_stddev = 0.0; for (const auto& pt : seed_points) h_stddev += std::pow(pt.dot(h_axis) - h_mean, 2);
        h_stddev = std::sqrt(h_stddev / static_cast<double>(seed_points.size()));
        double area = CalculateConvexHullPerimeter(seed_points);

        if (!(h_stddev < 5.0 && area > 900.0)) return {};

        std::cout << "  - 检测到地面 (" << (is_max_end ? "顶端" : "底端") << ")，正在移除..." << std::endl;

        // 拟合平面
        Eigen::Vector3d plane_centroid = Eigen::Vector3d::Zero();
        for (const auto& pt : seed_points) plane_centroid += pt;
        plane_centroid /= static_cast<double>(seed_points.size());

        Eigen::Matrix3d plane_cov = Eigen::Matrix3d::Zero();
        for (const auto& pt : seed_points) plane_cov += (pt - plane_centroid) * (pt - plane_centroid).transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> plane_solver(plane_cov);
        Eigen::Vector3d normal = plane_solver.eigenvectors().col(0);
        if (normal.dot(h_axis) < 0) normal = -normal;

        Eigen::Vector4d plane_model(normal.x(), normal.y(), normal.z(), -normal.dot(plane_centroid));
        std::vector<size_t> ground_indices;
        for (size_t i = 0; i < vertices.size(); ++i) {
            double dist = std::abs(plane_model.dot(Eigen::Vector4d(vertices[i].x(), vertices[i].y(), vertices[i].z(), 1.0)));
            if (dist < 15.0) ground_indices.push_back(i);
        }
        return ground_indices;
        };

    auto g_min = analyze_end(*mesh, pca1_height_axis, false);
    auto g_max = analyze_end(*mesh, pca1_height_axis, true);

    bool ground_detected = !g_min.empty() || !g_max.empty();
    Eigen::Vector3d final_height_axis, final_width_axis, final_depth_axis;

    if (ground_detected) {
        auto clean_mesh = std::make_shared<geometry::TriangleMesh>(*mesh);
        if (!g_min.empty()) clean_mesh->RemoveVerticesByIndex(g_min);
        if (!g_max.empty()) clean_mesh->RemoveVerticesByIndex(g_max);
        clean_mesh->RemoveUnreferencedVertices();
        mesh = clean_mesh;

        Eigen::Vector3d clean_centroid = mesh->GetCenter();
        Eigen::Matrix3d cov2 = Eigen::Matrix3d::Zero();
        for (const auto& v : mesh->vertices_) cov2 += (v - clean_centroid) * (v - clean_centroid).transpose();
        if (mesh->vertices_.size() > 1) cov2 /= static_cast<double>(mesh->vertices_.size() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es2(cov2);

        final_height_axis = es2.eigenvectors().col(2);
        final_width_axis = es2.eigenvectors().col(1);
        final_depth_axis = es2.eigenvectors().col(0);

        double dot = pca1_height_axis.dot(final_height_axis);
        if (std::abs(dot) > 0.95) {
            if (dot > 0) { if (!g_max.empty()) final_height_axis = -final_height_axis; }
            else { if (!g_min.empty()) final_height_axis = -final_height_axis; }
        }
    }
    else {
        std::cout << "  - 未检测到地面，使用原始 PCA。" << std::endl;
        final_height_axis = pca1_height_axis;
        final_width_axis = pca1_width_axis;
        final_depth_axis = pca1_principal_axes.col(0);
    }

    // 3. 确定 Z 轴
    auto temp_mesh = std::make_shared<geometry::TriangleMesh>(*mesh);
    Eigen::Matrix3d R_temp;
    R_temp.col(1) = final_height_axis.normalized();
    R_temp.col(2) = final_depth_axis.normalized();
    R_temp.col(0) = -R_temp.col(1).cross(R_temp.col(2)).normalized();
    temp_mesh->Transform(Eigen::Affine3d(R_temp.transpose()).matrix());

    auto bbox = temp_mesh->GetAxisAlignedBoundingBox();
    double min_y = bbox.GetMinBound().y(), h = bbox.GetExtent().y();
    std::vector<Eigen::Vector3d> l_prof, r_prof;
    for (double y = min_y + h * 0.80; y < min_y + h * 0.96; y += 5.0) {
        auto slice = GetSlice(y, 4.0, *temp_mesh);
        if (slice.size() < 2) continue;
        auto x_bounds = std::minmax_element(slice.begin(), slice.end(), [](const auto& a, const auto& b) { return a.x() < b.x(); });
        l_prof.push_back(*x_bounds.first); r_prof.push_back(*x_bounds.second);
    }

    if (l_prof.size() > 1 && r_prof.size() > 1) {
        double neck_base_y = (l_prof.front().y() + r_prof.front().y()) / 2.0;
        std::vector<Eigen::Vector2d> f_prof_2d, b_prof_2d;
        for (double y = neck_base_y; y < neck_base_y + 30.0; y += 3.0) {
            auto slice = GetSlice(y, 3.0, *temp_mesh);
            if (slice.size() < 2) continue;
            auto z_bounds = std::minmax_element(slice.begin(), slice.end(), [](const auto& a, const auto& b) { return a.z() < b.z(); });
            b_prof_2d.emplace_back(z_bounds.first->y(), z_bounds.first->z());
            f_prof_2d.emplace_back(z_bounds.second->y(), z_bounds.second->z());
        }
        if (f_prof_2d.size() > 2 && b_prof_2d.size() > 2) {
            auto fit_res_f = FitLine2D(f_prof_2d);
            auto fit_res_b = FitLine2D(b_prof_2d);
            double avg_slope = (fit_res_f.first + fit_res_b.first) / 2.0;
            if (avg_slope < -0.1) {
                std::cout << "  - Z轴反向修正 (胸部朝前)。" << std::endl;
                final_depth_axis = -final_depth_axis;
            }
        }
    }

    // 4. 应用变换
    Eigen::Vector3d final_Y = final_height_axis.normalized();
    Eigen::Vector3d final_Z = final_depth_axis.normalized();
    Eigen::Vector3d final_X = -final_Y.cross(final_Z).normalized();
    Eigen::Matrix3d final_R;
    final_R.col(0) = final_X; final_R.col(1) = final_Y; final_R.col(2) = final_Z;
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.rotate(final_R.transpose());
    T.pretranslate(-mesh->GetCenter());
    mesh->Transform(T.matrix());

    // 5. 【关键】最后一步：检查形态学，如果倒立则翻转
    CheckAndFlipHuman(mesh);

    // 6. 最终落地
    auto final_bbox = mesh->GetAxisAlignedBoundingBox();
    mesh->Translate(Eigen::Vector3d(0, -final_bbox.GetMinBound().y(), 0));
    std::cout << ">>> 预处理完成，模型已归一化。" << std::endl;
}

// =========================================================
// 4. Main 函数
// =========================================================

int main() {
    system("chcp 65001 > nul");

    // 【请修改路径】
    std::string input_path = "D:/work/C++/my_cpp_project/Fusion_C_308.stl";
    std::string output_path = "D:/work/C++/my_cpp_project/input2.obj";

    std::cout << "正在读取: " << input_path << std::endl;

    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (!io::ReadTriangleMesh(input_path, *mesh)) {
        std::cerr << "错误: 无法读取文件 " << input_path << std::endl;
        std::cerr << "请检查路径是否正确！" << std::endl;
        return -1;
    }

    try {
        FixOrientationAndNormalize(mesh);
    }
    catch (const std::exception& e) {
        std::cerr << "处理过程中出错: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "正在保存结果到: " << output_path << std::endl;
    if (io::WriteTriangleMesh(output_path, *mesh)) {
        std::cout << "✅ 成功生成 input.obj" << std::endl;
    }
    else {
        std::cerr << "❌ 保存失败！" << std::endl;
    }

    return 0;
}