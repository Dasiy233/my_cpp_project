// ============================================================================
// SMPL Fitting + 14 Measurements (V27.20 Final Integration)
// 包含：
// 1. [保留] 用户的原始 650次 3阶段优化循环 (解决骨骼位置对齐问题)
// 2. [升级] 智能躯干分离 (GetTorsoSlice) + 切片点云可视化
// 3. [配置] 最后在原始扫描模型上进行测量
// ============================================================================

#define NOMINMAX
#define _USE_MATH_DEFINES

#include <torch/torch.h>
#include <torch/script.h>

#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/geometry/KDTreeFlann.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

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
#include <cfloat>

using namespace torch::indexing;
using namespace open3d;
namespace fs = std::filesystem;

// ============================================================================
// 【配置区域】
// ============================================================================
const bool FIX_UPSIDE_DOWN = false;
const bool FIX_FRONT_BACK = false;
const float FORCE_ROTATION_ANGLE = 0.0f;

const std::string BASE_DIR = "D:/work/C++/my_cpp_project/";
const std::string MODEL_PATH = BASE_DIR + "smpl_male_30.pt";
const std::string SCAN_PATH = BASE_DIR + "input2.obj";
const std::string FACES_PATH = BASE_DIR + "smpl_faces.txt";

const std::string OUTPUT_DIR = BASE_DIR + "results2/";
const std::string OUTPUT_OBJ = OUTPUT_DIR + "output_smpl_highres.glb";
const std::string OUTPUT_COMPARISON = OUTPUT_DIR + "result_comparison.glb";
const std::string OUTPUT_JOINTS_ONLY = OUTPUT_DIR + "joints_on_input.glb";
const std::string OUTPUT_MERGED = OUTPUT_DIR + "model_with_lines.glb";

// 训练参数 (保持你的原样)
const float INIT_Y_OFFSET = 0.05f;
const int   NUM_BETAS = 30;
const int   TOTAL_ITERS = 650;
const float INIT_ARM_ANGLE = 0.8f;//标准 A-Pose	最常见的扫描姿势，手臂自然张开	
//0.6f ~ 0.9f	35° ~ 50°
//窄 A - Pose(当前设置)	手臂比较垂，稍微离开身体	1.0f

// ============================================================================
// 【辅助函数】(优化循环依赖这些函数)
// ============================================================================
double dist_v3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return (a - b).norm(); }
static inline double clampd(double x, double a, double b) { return std::max(a, std::min(b, x)); }

// Torch Tensor 设置辅助
static inline void SetTensorAt2D(torch::Tensor& t, int row, int col, float v) {
    auto sv = torch::full({}, v, t.options());
    t.index_put_({ row, col }, sv);
}

// 点到线段距离 (用于 Loss 计算)
torch::Tensor point_segment_distance(torch::Tensor p, torch::Tensor a, torch::Tensor b) {
    auto ab = b - a;
    auto denom = torch::sum(ab * ab) + 1e-8;
    auto t = torch::sum((p - a) * ab) / denom;
    t = torch::clamp(t, 0.0, 1.0);
    auto proj = a + t * ab;
    return torch::norm(p - proj);
}

// Chamfer Distance (用于 Loss 计算)
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

// 关节可视化 (修复 Accessor 报错)
// ===== SMPL 24 joints 名字 & 颜色（0~1）=====
static const char* JOINT_NAMES[24] = {
    "pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2","l_ankle","r_ankle","spine3",
    "l_foot","r_foot","neck","l_collar","r_collar","head","l_shoulder","r_shoulder","l_elbow","r_elbow",
    "l_wrist","r_wrist","l_hand","r_hand"
};

// 24个高区分度颜色（你也可以自己换）
static const Eigen::Vector3d JOINT_COLORS[24] = {
    {1.00,0.00,0.00},{0.00,1.00,0.00},{0.00,0.00,1.00},{1.00,1.00,0.00},
    {1.00,0.00,1.00},{0.00,1.00,1.00},{1.00,0.50,0.00},{0.50,0.00,1.00},
    {0.00,0.50,1.00},{0.00,0.80,0.20},{0.80,0.20,0.00},{0.20,0.20,0.20},
    {0.60,0.60,0.00},{0.60,0.00,0.60},{0.00,0.60,0.60},{0.90,0.30,0.40},
    {0.40,0.90,0.30},{0.30,0.40,0.90},{0.90,0.70,0.20},{0.20,0.90,0.70},
    {0.70,0.20,0.90},{0.90,0.20,0.70},{0.20,0.70,0.90},{0.70,0.90,0.20},
};

// ===== 关节可视化：每个 joint 不同颜色 + 控制台打印对照表 =====
std::shared_ptr<geometry::TriangleMesh> create_joints_visual(const torch::Tensor& joints, double sphere_r = 0.018) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    torch::Tensor tmp = joints.squeeze(0).cpu();
    auto acc = tmp.accessor<float, 2>();

    std::cout << "\n[SMPL 24 joints legend]\n";
    for (int i = 0; i < 24; ++i) {
        auto s = geometry::TriangleMesh::CreateSphere(sphere_r, 10);
        s->Translate(Eigen::Vector3d(acc[i][0], acc[i][1], acc[i][2]));
        s->PaintUniformColor(JOINT_COLORS[i]);  // 每个关节不同颜色
        *mesh += *s;

        const auto& c = JOINT_COLORS[i];
        std::cout << i << "  " << JOINT_NAMES[i]
            << "  color=(" << c.x() << "," << c.y() << "," << c.z() << ")\n";
    }
    return mesh;
}


// ============================================================================
// 【核心几何算法：切片、清洗、凸包】
// ============================================================================

// 1. 获取切片 (Y轴)
static std::vector<Eigen::Vector3d> GetSlice(const open3d::geometry::TriangleMesh& mesh,
    double y_height, double tolerance) {
    std::vector<Eigen::Vector3d> slice_points;
    const double min_y = y_height - tolerance / 2.0;
    const double max_y = y_height + tolerance / 2.0;
    for (const auto& vertex : mesh.vertices_) {
        if (vertex.y() >= min_y && vertex.y() <= max_y) slice_points.push_back(vertex);
    }
    return slice_points;
}

// 2. 获取切片 (X轴, 用于袖笼)
static std::vector<Eigen::Vector3d> GetSliceX(const open3d::geometry::TriangleMesh& mesh, double x_val, double tolerance) {
    std::vector<Eigen::Vector3d> slice_points;
    const double min_x = x_val - tolerance / 2.0;
    const double max_x = x_val + tolerance / 2.0;
    for (const auto& vertex : mesh.vertices_) {
        if (vertex.x() >= min_x && vertex.x() <= max_x) slice_points.push_back(vertex);
    }
    return slice_points;
}

// 3. 智能躯干分离器 (整数索引版 - 修复精度问题)
static std::vector<Eigen::Vector3d> GetTorsoSlice(
    const std::vector<Eigen::Vector3d>& full_slice_pts,
    double eps = 0.015,
    int min_points = 10,
    double thickness_ratio = 0.65,
    bool debug = false)
{
    if (full_slice_pts.size() < min_points * 2) return full_slice_pts;

    // DBSCAN
    auto pcd = std::make_shared<geometry::PointCloud>();
    pcd->points_ = full_slice_pts;
    std::vector<int> labels = pcd->ClusterDBSCAN(eps, min_points, false);

    std::map<int, int> counts;
    for (int label : labels) if (label != -1) counts[label]++;

    if (debug) std::cout << "\n[DEBUG] DBSCAN Clusters: " << counts.size() << " (eps=" << eps << ")" << std::endl;
    if (debug) {
        std::vector<std::pair<int, int>> items(counts.begin(), counts.end());
        std::sort(items.begin(), items.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        for (auto& it : items) {
            std::cout << "  - label " << it.first << " : " << it.second << " pts" << std::endl;
        }
    }

    if (counts.size() > 1) {
        if (debug) std::cout << "  -> 聚类成功分离，取最大簇。" << std::endl;
        int max_label = -1, max_c = -1;
        for (auto const& [l, c] : counts) {
            if (c > max_c) { max_c = c; max_label = l; }
        }
        std::vector<Eigen::Vector3d> torso;
        for (size_t i = 0; i < full_slice_pts.size(); ++i) {
            if (labels[i] == max_label) torso.push_back(full_slice_pts[i]);
        }
        return torso;
    }

    // 深度分析 (Integer Binning)
    if (debug) std::cout << "  -> 启动深度分析 (粘连处理)..." << std::endl;
    auto sorted = full_slice_pts;
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.x() < b.x(); });

    double min_x = sorted.front().x();
    double max_x = sorted.back().x();
    double step = 0.005;
    int num_bins = std::floor((max_x - min_x) / step) + 2;

    std::vector<double> min_z(num_bins, 1e9);
    std::vector<double> max_z(num_bins, -1e9);
    std::vector<bool> has_data(num_bins, false);

    for (const auto& p : sorted) {
        int idx = std::floor((p.x() - min_x) / step);
        if (idx >= 0 && idx < num_bins) {
            if (p.z() < min_z[idx]) min_z[idx] = p.z();
            if (p.z() > max_z[idx]) max_z[idx] = p.z();
            has_data[idx] = true;
        }
    }

    double max_thickness = 0.0;
    int peak_idx = 0;
    std::vector<double> thickness(num_bins, 0.0);

    for (int i = 0; i < num_bins; ++i) {
        if (has_data[i]) {
            double t = max_z[i] - min_z[i];
            thickness[i] = t;
            if (t > max_thickness) { max_thickness = t; peak_idx = i; }
        }
    }

    double depth_threshold = max_thickness * thickness_ratio;

    if (debug) {
        std::cout << "  -> 最大厚度: " << max_thickness << " m" << std::endl;
        std::cout << "  -> 切割阈值: " << depth_threshold << " m (Ratio: " << thickness_ratio << ")" << std::endl;
    }

    int min_valid_idx = 0;
    int max_valid_idx = num_bins - 1;

    for (int i = peak_idx; i >= 0; --i) {
        if (has_data[i] && thickness[i] < depth_threshold) {
            min_valid_idx = i;
            if (debug) std::cout << "  -> 切割左界: Index " << i << " (Thick=" << thickness[i] << ")" << std::endl;
            break;
        }
    }
    for (int i = peak_idx; i < num_bins; ++i) {
        if (has_data[i] && thickness[i] < depth_threshold) {
            max_valid_idx = i;
            if (debug) std::cout << "  -> 切割右界: Index " << i << " (Thick=" << thickness[i] << ")" << std::endl;
            break;
        }
    }

    std::vector<Eigen::Vector3d> result;
    for (const auto& p : full_slice_pts) {
        int idx = std::floor((p.x() - min_x) / step);
        if (idx > min_valid_idx && idx < max_valid_idx) {
            result.push_back(p);
        }
    }

    if (debug) std::cout << "  -> 结果: " << full_slice_pts.size() << " -> " << result.size() << " pts\n";
    if (result.size() < min_points) return full_slice_pts;
    return result;
}

// 4. 二维凸包 (Andrew's Monotone Chain)
static double cross2(const Eigen::Vector2d& O, const Eigen::Vector2d& A, const Eigen::Vector2d& B) {
    return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
}
std::vector<Eigen::Vector3d> GetConvexHullPointsXZ(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) return points;
    struct P { Eigen::Vector2d p; size_t idx; };
    std::vector<P> pts;
    pts.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) pts.push_back({ Eigen::Vector2d(points[i].x(), points[i].z()), i });

    std::sort(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.p.x() < b.p.x() || (a.p.x() == b.p.x() && a.p.y() < b.p.y());
        });

    std::vector<size_t> H;
    for (size_t i = 0; i < pts.size(); ++i) {
        while (H.size() >= 2) {
            if (cross2(pts[H[H.size() - 2]].p, pts[H.back()].p, pts[i].p) <= 0) H.pop_back();
            else break;
        }
        H.push_back(i);
    }
    size_t lower_size = H.size();
    for (int i = (int)pts.size() - 2; i >= 0; --i) {
        while (H.size() > lower_size) {
            if (cross2(pts[H[H.size() - 2]].p, pts[H.back()].p, pts[i].p) <= 0) H.pop_back();
            else break;
        }
        H.push_back((size_t)i);
    }
    if (H.size() > 1) H.pop_back();

    std::vector<Eigen::Vector3d> hull;
    for (auto h : H) hull.push_back(points[pts[h].idx]);
    return hull;
}

// 5. 重采样
static std::vector<Eigen::Vector3d> ResampleClosedRing(const std::vector<Eigen::Vector3d>& ring, int N) {
    std::vector<Eigen::Vector3d> out;
    if (ring.size() < 2 || N < 2) return out;
    std::vector<double> acc(ring.size() + 1, 0.0);
    for (size_t i = 0; i < ring.size(); ++i) acc[i + 1] = acc[i] + (ring[(i + 1) % ring.size()] - ring[i]).norm();
    double L = acc.back();
    if (L < 1e-9) return out;

    for (int k = 0; k < N; ++k) {
        double s = (L * k) / (double)N;
        size_t seg = 0;
        while (seg + 1 < acc.size() && acc[seg + 1] < s) ++seg;
        size_t i0 = seg % ring.size();
        size_t i1 = (i0 + 1) % ring.size();
        double segL = (ring[i1] - ring[i0]).norm();
        double t = (segL < 1e-9) ? 0.0 : (s - acc[seg]) / segL;
        out.push_back(ring[i0] * (1.0 - t) + ring[i1] * t);
    }
    return out;
}

// 6. 周长
static double PerimeterClosed(const std::vector<Eigen::Vector3d>& ring) {
    if (ring.size() < 2) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < ring.size(); ++i) sum += (ring[(i + 1) % ring.size()] - ring[i]).norm();
    return sum;
}

// 7. 可视化 (显示切片点云 -> 小球)
static std::shared_ptr<geometry::TriangleMesh> VisualizeSlicePoints(const std::vector<Eigen::Vector3d>& points, const Eigen::Vector3d& color, double radius = 0.004) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (points.empty()) return mesh;
    auto base_sphere = geometry::TriangleMesh::CreateSphere(radius, 6); // Low poly sphere
    base_sphere->PaintUniformColor(color);
    for (const auto& p : points) {
        auto s = std::make_shared<geometry::TriangleMesh>(*base_sphere);
        s->Translate(p);
        *mesh += *s;
    }
    return mesh;
}
static std::shared_ptr<geometry::TriangleMesh> VisualizePathTube(const std::vector<Eigen::Vector3d>& path,
    const Eigen::Vector3d& color, double r = 0.002) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();

    if (path.size() < 2) return mesh;
    for (size_t i = 0; i < path.size(); ++i) {
        auto p1 = path[i];
        auto p2 = path[(i + 1) % path.size()];
        auto v = p2 - p1;
        double len = v.norm();
        if (len < 1e-9) continue;
        auto cyl = geometry::TriangleMesh::CreateCylinder(r, len, 8);
        cyl->PaintUniformColor(color);
        Eigen::Vector3d dir = v / len;
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(0, 0, 1), dir);
        cyl->Rotate(q.toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
        cyl->Translate((p1 + p2) * 0.5);
        *mesh += *cyl;
    }
    return mesh;
}
static std::shared_ptr<geometry::TriangleMesh> VisualizePath(const std::vector<Eigen::Vector3d>& p, const Eigen::Vector3d& c) {
    return VisualizePathTube(p, c);
}

static double DistPointSegment(const Eigen::Vector3d& p,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b) {
    Eigen::Vector3d ab = b - a;
    double t = (p - a).dot(ab) / (ab.dot(ab) + 1e-12);
    t = std::max(0.0, std::min(1.0, t));
    Eigen::Vector3d proj = a + t * ab;
    return (p - proj).norm();
}

static std::vector<Eigen::Vector3d> RemoveArmsByJoints(
    const std::vector<Eigen::Vector3d>& pts,
    const std::vector<Eigen::Vector3d>& J,
    double r = 0.06   // 6cm，通常 0.05~0.08 之间调
) {
    const int LS = 16, LE = 18, LW = 20;
    const int RS = 17, RE = 19, RW = 21;

    std::vector<Eigen::Vector3d> out;
    out.reserve(pts.size());
    for (auto& p : pts) {
        double dl = std::min(DistPointSegment(p, J[LS], J[LE]), DistPointSegment(p, J[LE], J[LW]));
        double dr = std::min(DistPointSegment(p, J[RS], J[RE]), DistPointSegment(p, J[RE], J[RW]));
        if (dl > r && dr > r) out.push_back(p);
    }
    return out;
}



// ============================================================================
// 【测量主逻辑】 v27.20 智能测量
// ============================================================================
void measure_14_items(std::shared_ptr<geometry::TriangleMesh>& mesh, const torch::Tensor& joints_tensor) {
    const double VIS_TOL_Y = 0.01;      // Y切片可视化厚度
    const double VIS_TOL_X = 0.03;      // X切片可视化厚度（袖笼）
    const double VIS_SPHERE_R = 0.0008; // 点球半径

    torch::Tensor tmp = joints_tensor.squeeze(0).cpu();
    auto joints_acc = tmp.accessor<float, 2>();
    std::vector<Eigen::Vector3d> J(24);
    for (int i = 0; i < 24; ++i) J[i] = Eigen::Vector3d(joints_acc[i][0], joints_acc[i][1], joints_acc[i][2]);

    auto bbox = mesh->GetAxisAlignedBoundingBox();
    double min_y = bbox.GetMinBound().y();
    double height = bbox.GetMaxBound().y() - min_y;
    auto all_visuals = std::make_shared<geometry::TriangleMesh>();



    std::cout << "\n================ [ 14项专业身体测量报告 ] ================" << std::endl;
    std::cout << "身高            : " << height * 100.0 << " cm" << std::endl;



    // 1) 颈围
    double neck_y = J[12].y() + 0.05;
    auto neck_raw = GetSlice(*mesh, neck_y, 0.015);
    auto neck_clean = GetTorsoSlice(neck_raw, 0.02, 5, 0.50);
    // 可视化：只画原始切片点云（不再用clean，避免点太少）
    if (!neck_raw.empty()) *all_visuals += *VisualizeSlicePoints(neck_raw, { 1,1,0 }, VIS_SPHERE_R);
    auto neck_hull = GetConvexHullPointsXZ(neck_clean);
    auto neck_ring = ResampleClosedRing(neck_hull, 120);
    std::cout << "1)  颈围        : " << PerimeterClosed(neck_ring) * 100.0 << " cm" << std::endl;


    // 2) 胸围（不做手臂剔除：DBSCAN 出 3 簇，取最大簇=躯干）
    double chest_y = J[6].y() * 0.15 + J[9].y() * 0.85;
    // 测量用厚切片（tol=0.02 固定）
    auto chest_raw = GetSlice(*mesh, chest_y, 0.02);

    // 固定你刚验证成功的参数：eps=0.008, minPts=20
    auto chest_clean = GetTorsoSlice(chest_raw, 0.008, 20, 0.70, TRUE);

    // 可视化：只画躯干簇（不显示手臂）
// 可视化：直接画测量得到的躯干簇（稳定、不会被薄切片打碎）
    if (!chest_clean.empty())
        *all_visuals += *VisualizeSlicePoints(chest_clean, { 0,1,0 }, VIS_SPHERE_R);


    // 兜底：万一某次 clean 为空，至少不要崩
    if (chest_clean.empty()) chest_clean = chest_raw;

    auto chest_hull = GetConvexHullPointsXZ(chest_clean);
    auto chest_ring = ResampleClosedRing(chest_hull, 240);
    // 调试一下有没有生成ring
    std::cout << "[chest] raw=" << chest_raw.size()
        << " clean=" << chest_clean.size()
        << " hull=" << chest_hull.size()
        << " ring=" << chest_ring.size() << "\n";



    std::cout << "2)  胸围        : " << PerimeterClosed(chest_ring) * 100.0 << " cm" << std::endl;


    // 3) 腰围
    double waist_y = J[3].y();
    auto waist_raw = GetSlice(*mesh, waist_y, 0.02);
    auto waist_clean = GetTorsoSlice(waist_raw, 0.02, 10, 0.55);
    // 可视化：只画原始切片点云
    auto waist_vis_raw = GetSlice(*mesh, waist_y, VIS_TOL_Y);
    auto waist_vis_torso = GetTorsoSlice(waist_vis_raw, 0.02, 10, 0.55, false);
    if (!waist_vis_torso.empty())
        *all_visuals += *VisualizeSlicePoints(waist_vis_torso, { 1,0,1 }, VIS_SPHERE_R);

    auto waist_hull = GetConvexHullPointsXZ(waist_clean);
    auto waist_ring = ResampleClosedRing(waist_hull, 200);
    double waist_girth = PerimeterClosed(waist_ring);

    double max_waist_x = -1e9;
    if (!waist_ring.empty()) for (auto& p : waist_ring) if (p.x() > max_waist_x) max_waist_x = p.x();
    else max_waist_x = J[0].x() + 0.15;

    std::cout << "3)  腰围        : " << waist_girth * 100.0 << " cm" << std::endl;


    // 4) 臀围
    double hip_best = 0.0;
    double hip_best_y = J[0].y();
    std::vector<Eigen::Vector3d> hip_clean_best;
    for (double y = J[0].y() - 0.12; y < J[0].y() + 0.04; y += 0.01) {
        auto raw = GetSlice(*mesh, y, 0.003);
        if (raw.empty()) continue;
        auto clean = GetTorsoSlice(raw, 0.015, 10, 0.55);
        if (clean.size() < 20) continue;
        auto hull = GetConvexHullPointsXZ(clean);
        auto ring = ResampleClosedRing(hull, 240);
        double per = PerimeterClosed(ring);
        if (per > 0.60 && per < 1.30 && per > hip_best) {
            hip_best = per; hip_clean_best = clean; hip_best_y = y;
        }
    }
    {
        auto hip_vis_raw = GetSlice(*mesh, hip_best_y, VIS_TOL_Y);
        auto hip_vis_torso = GetTorsoSlice(hip_vis_raw, 0.02, 10, 0.65, false); // 只用于可视化
        if (!hip_vis_torso.empty())
            *all_visuals += *VisualizeSlicePoints(hip_vis_torso, { 1,0.5,0 }, VIS_SPHERE_R);

    }
    std::cout << "4)  臀围        : " << hip_best * 100.0 << " cm" << std::endl;

    // 5) 大腿围
    int L_HIP = 1, R_HIP = 2, L_KNEE = 4;
    double thigh_y0 = 0.60 * J[L_HIP].y() + 0.40 * J[L_KNEE].y();
    double thigh_best = 0.0;
    double thigh_best_y = thigh_y0;
    std::vector<Eigen::Vector3d> thigh_clean_best;
    for (double y = thigh_y0 + 0.04; y >= thigh_y0 - 0.06; y -= 0.01) {
        auto raw = GetSlice(*mesh, y, 0.02);
        if (raw.empty()) continue;
        std::vector<Eigen::Vector3d> left_leg;
        for (auto& p : raw) if ((p - J[L_HIP]).norm() < (p - J[R_HIP]).norm() - 0.005) left_leg.push_back(p);

        auto clean = GetTorsoSlice(left_leg, 0.015, 10, 0.60);
        if (clean.size() < 20) continue;
        auto hull = GetConvexHullPointsXZ(clean);
        auto ring = ResampleClosedRing(hull, 200);
        double per = PerimeterClosed(ring);
        if (per > 0.30 && per < 0.90 && per > thigh_best) {
            thigh_best = per; thigh_clean_best = clean; thigh_best_y = y;
        }
    }
    {
        // 可视化：原始点云（左腿）-> DBSCAN -> 取最大簇（大腿）
        const double VIS_THIGH_TOL = 0.01;   // 画图用切片厚度：越小越细（0.008~0.012）
        const double EPS = 0.015;           // 跟你测量循环里一致
        const int    MINP = 10;

        auto thigh_vis_raw = GetSlice(*mesh, thigh_best_y, VIS_THIGH_TOL);

        // 先取左侧（原始点云，不做其它过滤）
        std::vector<Eigen::Vector3d> thigh_vis_left;
        thigh_vis_left.reserve(thigh_vis_raw.size());
        for (auto& p : thigh_vis_raw) {
            if ((p - J[L_HIP]).norm() < (p - J[R_HIP]).norm() - 0.005)
                thigh_vis_left.push_back(p);
        }

        // DBSCAN：取最大簇
        std::vector<Eigen::Vector3d> thigh_vis_big;
        if (!thigh_vis_left.empty()) {
            auto pcd = std::make_shared<open3d::geometry::PointCloud>();
            pcd->points_ = thigh_vis_left;

            auto labels = pcd->ClusterDBSCAN(EPS, MINP, false);

            std::map<int, int> cnt;
            for (int lb : labels) if (lb != -1) cnt[lb]++;

            int best_label = -1, best_sz = 0;
            for (auto& kv : cnt) {
                if (kv.second > best_sz) { best_sz = kv.second; best_label = kv.first; }
            }

            if (best_label != -1) {
                thigh_vis_big.reserve(best_sz);
                for (size_t i = 0; i < thigh_vis_left.size(); ++i) {
                    if (labels[i] == best_label) thigh_vis_big.push_back(thigh_vis_left[i]);
                }
            }

            std::cout << "[thigh_vis] left=" << thigh_vis_left.size()
                << " clusters=" << cnt.size()
                << " best=" << best_sz
                << " (eps=" << EPS << ",minPts=" << MINP << ")\n";
        }

        // 画“最大簇”的原始点云（这就是你要的大腿部分）
        if (!thigh_vis_big.empty())
            *all_visuals += *VisualizeSlicePoints(thigh_vis_big, { 1,0,0 }, VIS_SPHERE_R);

        // 画线：用最大簇的凸包 ring（只会在大腿处）
        if (!thigh_vis_big.empty()) {
            auto hull = GetConvexHullPointsXZ(thigh_vis_big);
            auto ring = ResampleClosedRing(hull, 200);
            if (ring.size() >= 3) {
                for (auto& p : ring) p.y() = thigh_best_y;
                *all_visuals += *VisualizePathTube(ring, { 1,0,0 }, 0.0010); // 想更细：0.0008
            }
        }
    }

    std::cout << "5)  大腿围      : " << thigh_best * 100.0 << " cm" << std::endl;


    // 6) 肩宽
    double sh_dist = dist_v3(J[16], J[17]);
    std::cout << "6)  肩宽        : " << (sh_dist + 0.12) * 100.0 << " cm" << std::endl;

    // 7) 袖笼围
    {
        double x_cut = J[16].x() + 0.02;
        auto arm_raw = GetSliceX(*mesh, x_cut, VIS_TOL_X);
        std::vector<Eigen::Vector3d> arm_clean, points_mapped, ring;
        for (auto& p : arm_raw) if ((p - J[16]).norm() < 0.25) arm_clean.push_back(p);
        if (!arm_clean.empty()) *all_visuals += *VisualizeSlicePoints(arm_clean, { 0,0,1 }, VIS_SPHERE_R);

        for (auto& p : arm_clean) points_mapped.push_back({ p.y(), 0, p.z() });
        auto hull = GetConvexHullPointsXZ(points_mapped);
        for (auto& p : hull) ring.push_back({ x_cut, p.x(), p.z() });
        ring = ResampleClosedRing(ring, 120);
        double armhole = PerimeterClosed(ring);
        std::cout << "7)  袖笼围      : " << armhole * 100.0 << " cm" << std::endl;
    }

    // 8-14
    double arm_len = dist_v3(J[16], J[18]) + dist_v3(J[18], J[20]);
    std::cout << "8)  臂长        : " << (arm_len + 0.05) * 100.0 << " cm" << std::endl;
    std::cout << "9)  中指长      : " << height * 0.11 * 0.45 * 100.0 << " cm (AI)" << std::endl;
    std::cout << "10) 手长        : " << height * 0.11 * 100.0 << " cm (AI)" << std::endl;
    std::cout << "11) 手宽        : " << height * 0.11 * 0.48 * 100.0 << " cm (AI)" << std::endl;
    std::cout << "12) 躯干垂直围  : " << ((J[12].y() - J[0].y()) * 2.0 + 0.15) * 100.0 << " cm" << std::endl;
    std::cout << "13) 衣长        : " << (J[12].y() - J[1].y()) * 100.0 << " cm" << std::endl;

    double pants_len = (waist_girth > 0) ? std::max(0.0, (waist_y - min_y) - 0.03) : 0.0;
    std::cout << "14) 裤长        : " << pants_len * 100.0 << " cm" << std::endl;

    if (pants_len > 0) {
        auto cyl = geometry::TriangleMesh::CreateCylinder(0.008, pants_len, 12);
        cyl->PaintUniformColor({ 0.5, 0, 0.5 });
        cyl->Translate({ max_waist_x + 0.05, waist_y - pants_len / 2.0, 0 });
        Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
        cyl->Rotate(R, { max_waist_x + 0.05, waist_y - pants_len / 2.0, 0 });
        *all_visuals += *cyl;
    }

    *mesh += *all_visuals;
    io::WriteTriangleMesh(OUTPUT_MERGED, *mesh);
    std::cout << "✅ 测量结束，模型(含点云)已保存: " << OUTPUT_MERGED << std::endl;
}

// ============================================================================
// 【主函数】
// ============================================================================
int main() {
    system("chcp 65001 > nul");
    try {
        if (!fs::exists(OUTPUT_DIR)) fs::create_directories(OUTPUT_DIR);

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << ">>> 🚀 使用 GPU 加速" << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        // 1. 加载 SMPL
        torch::jit::script::Module smpl_module = torch::jit::load(MODEL_PATH);
        smpl_module.to(device);

        // 2. 加载扫描并预处理
        std::cout << ">>> 加载扫描: " << SCAN_PATH << std::endl;
        auto target_ptr = io::CreateMeshFromFile(SCAN_PATH);
        if (!target_ptr || target_ptr->vertices_.empty()) throw std::runtime_error("扫描为空");

        auto bbox = target_ptr->GetAxisAlignedBoundingBox();
        if (bbox.GetMaxBound().y() - bbox.GetMinBound().y() > 10.0) target_ptr->Scale(0.001, target_ptr->GetCenter());
        if (FIX_UPSIDE_DOWN) target_ptr->Rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix(), target_ptr->GetCenter());
        if (FIX_FRONT_BACK) target_ptr->Rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix(), target_ptr->GetCenter());
        target_ptr->Translate(-target_ptr->GetAxisAlignedBoundingBox().GetCenter());
        target_ptr->PaintUniformColor({ 0.6, 0.6, 0.6 });

        // 3. 采样点云 (用于拟合)
        auto pcd = target_ptr->SamplePointsUniformly(10000);
        int N = (int)pcd->points_.size();
        torch::Tensor target_verts_cpu = torch::zeros({ 1, N, 3 }, torch::kCPU);
        auto acc = target_verts_cpu.accessor<float, 3>();
        for (int i = 0; i < N; ++i) {
            acc[0][i][0] = pcd->points_[i].x();
            acc[0][i][1] = pcd->points_[i].y();
            acc[0][i][2] = pcd->points_[i].z();
        }
        torch::Tensor target_verts = target_verts_cpu.to(device);

        // 4. 【核心】原始优化循环 (Restored)
        auto betas = torch::zeros({ 1, NUM_BETAS }, torch::requires_grad().device(device));
        auto pose = torch::zeros({ 1, 69 }, torch::requires_grad().device(device));
        auto global_orient = torch::zeros({ 1, 3 }, torch::requires_grad().device(device));
        auto transl = torch::tensor({ 0.0f, INIT_Y_OFFSET, 0.0f }, device).reshape({ 1, 3 });
        transl.set_requires_grad(true);
        auto scale = torch::tensor({ 1.0 }, torch::requires_grad().device(device));

        {
            torch::NoGradGuard no_grad;
            SetTensorAt2D(pose, 0, 47, -std::abs(INIT_ARM_ANGLE));
            SetTensorAt2D(pose, 0, 50, std::abs(INIT_ARM_ANGLE));
        }
        std::cout << ">>> [手动模式] 强制设定初始朝向: " << int(FORCE_ROTATION_ANGLE * 180 / M_PI) << " 度" << std::endl;
        {
            torch::NoGradGuard no_grad;
            SetTensorAt2D(global_orient, 0, 1, FORCE_ROTATION_ANGLE);
        }

        std::vector<torch::Tensor> params = { transl, scale, betas, pose, global_orient };
        torch::optim::Adam optimizer(params, torch::optim::AdamOptions(0.02));

        std::cout << "\n>>> 开始优化 (结果将存入 " << OUTPUT_DIR << ")..." << std::endl;

        for (int i = 0; i < TOTAL_ITERS; ++i) {
            optimizer.zero_grad();

            float w_pose_reg = 0.0f;
            float w_beta_reg = 0.0f;
            std::string stage = "";
            bool lock_body_pose = false;
            bool lock_detail_betas = false;

            if (i < 150) {
                stage = "Stage 1: 刚性";
                for (auto& group : optimizer.param_groups())
                    static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.02);
                w_pose_reg = 10.0f; w_beta_reg = 0.0f;
                lock_body_pose = true; lock_detail_betas = true;

                // stage1 强制锁肩
                { torch::NoGradGuard ng; SetTensorAt2D(pose, 0, 47, -std::abs(INIT_ARM_ANGLE)); SetTensorAt2D(pose, 0, 50, std::abs(INIT_ARM_ANGLE)); }
            }
            else if (i < 450) {
                stage = "Stage 2: 体型";
                for (auto& group : optimizer.param_groups())
                    static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.05);
                w_pose_reg = 1.0f;
                w_beta_reg = 0.05f;
            }
            else {
                stage = "Stage 3: 微调";
                for (auto& group : optimizer.param_groups())
                    static_cast<torch::optim::AdamOptions&>(group.options()).lr(0.005);
                w_pose_reg = 0.1f; w_beta_reg = 0.0f;
            }

            std::vector<torch::jit::IValue> in;
            in.push_back(betas); in.push_back(pose); in.push_back(global_orient); in.push_back(transl); in.push_back(scale);
            auto out = smpl_module.forward(in).toTuple();
            auto smpl_verts = out->elements()[0].toTensor();
            auto smpl_joints = out->elements()[1].toTensor();

            auto loss_dist = simple_chamfer_distance(smpl_verts, target_verts);
            auto loss_reg_pose = torch::mean(torch::pow(pose, 2)) * w_pose_reg;
            auto loss_reg_beta = torch::mean(torch::pow(betas, 2)) * w_beta_reg;
            auto total_loss = loss_dist + loss_reg_pose + loss_reg_beta;

            total_loss.backward();

            if (lock_body_pose && pose.grad().defined())
                pose.grad().index({ Slice(), Slice(3, torch::indexing::None) }).fill_(0.0);
            if (lock_detail_betas && betas.grad().defined())
                betas.grad().index({ Slice(), Slice(10, torch::indexing::None) }).fill_(0.0);

            optimizer.step();

            if (i % 50 == 0) {
                auto J = smpl_joints.squeeze(0);
                auto Lw = J.index({ 20 });
                auto Rw = J.index({ 21 });
                auto Lhip = J.index({ 1 });
                auto Rhip = J.index({ 2 });
                auto Lk = J.index({ 4 });
                auto Rk = J.index({ 5 });
                auto Lsho = J.index({ 16 });
                auto Rsho = J.index({ 17 });
                auto Lel = J.index({ 18 });
                auto Rel = J.index({ 19 });

                float dL_leg = point_segment_distance(Lw, Lhip, Lk).item<float>();
                float dR_leg = point_segment_distance(Rw, Rhip, Rk).item<float>();
                float dL_arm = point_segment_distance(Lw, Lsho, Lel).item<float>();
                float dR_arm = point_segment_distance(Rw, Rsho, Rel).item<float>();

                float vL = pose.detach().index({ 0, 47 }).item<float>();
                float vR = pose.detach().index({ 0, 50 }).item<float>();

                std::cout << "Iter " << i << " [" << stage << "]"
                    << " | Dist: " << loss_dist.item<float>()
                    << " | sh=(" << vL << "," << vR << ")"
                    << " | dL_leg=" << dL_leg << " dR_leg=" << dR_leg
                    << " | dL_arm=" << dL_arm << " dR_arm=" << dR_arm
                    << std::endl;
            }
        }

        std::cout << "\n>>> 优化完成，正在保存..." << std::endl;

        // 5. 获取最终数据 & 保存
        std::vector<torch::jit::IValue> in;
        in.push_back(betas); in.push_back(pose); in.push_back(global_orient); in.push_back(transl); in.push_back(scale);
        auto outputs = smpl_module.forward(in).toTuple();

        // [Access Fix] 分两步获取 Accessor (避免 C2280 错误)
        torch::Tensor final_verts_t = outputs->elements()[0].toTensor().squeeze(0).cpu();
        auto final_verts_acc = final_verts_t.accessor<float, 2>();
        auto final_joints = outputs->elements()[1].toTensor();

        auto smpl_mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        for (int k = 0; k < final_verts_t.size(0); ++k)
            smpl_mesh_ptr->vertices_.push_back(Eigen::Vector3d(final_verts_acc[k][0], final_verts_acc[k][1], final_verts_acc[k][2]));

        if (load_faces(FACES_PATH, smpl_mesh_ptr->triangles_)) {
            smpl_mesh_ptr->ComputeVertexNormals();
            io::WriteTriangleMesh(OUTPUT_OBJ, *smpl_mesh_ptr);
        }

        // 保存比较模型
        auto comparison_mesh = std::make_shared<geometry::TriangleMesh>();
        *comparison_mesh += *target_ptr;
        // 创建一个临时的 SMPL 副本，涂成红色
        auto smpl_red = std::make_shared<geometry::TriangleMesh>(*smpl_mesh_ptr);
        smpl_red->PaintUniformColor({ 1.0, 0.0, 0.0 }); // RGB: 1,0,0 = 红色
        *comparison_mesh += *smpl_red;
        //*comparison_mesh += *smpl_mesh_ptr;
        io::WriteTriangleMesh(OUTPUT_COMPARISON, *comparison_mesh);

        // 保存关节叠加模型
        auto joints_mesh_ptr = create_joints_visual(final_joints, 0.018);
       
        auto scan_grey = std::make_shared<geometry::TriangleMesh>(*target_ptr);
        scan_grey->PaintUniformColor({ 0.65, 0.65, 0.65 });
        scan_grey->ComputeTriangleNormals();
        scan_grey->ComputeVertexNormals();

        auto joints_check_mesh = std::make_shared<geometry::TriangleMesh>();
        *joints_check_mesh += *scan_grey;
        *joints_check_mesh += *joints_mesh_ptr;

        joints_check_mesh->ComputeTriangleNormals();
        joints_check_mesh->ComputeVertexNormals();
        io::WriteTriangleMesh(OUTPUT_JOINTS_ONLY, *joints_check_mesh);


        // 3. 导出 smpl_with_joints (只展示 SMPL + 彩色关节球)
        {
            // 关键：给 SMPL 本体也写入 vertex_colors_，否则合并可能丢球的颜色
            auto smpl_body = std::make_shared<geometry::TriangleMesh>(*smpl_mesh_ptr);
            smpl_body->PaintUniformColor({ 0.75, 0.75, 0.75 }); // 灰色（随便）
            smpl_body->ComputeTriangleNormals();
            smpl_body->ComputeVertexNormals();

            // joints_mesh_ptr 本身已经是 24 色（create_joints_visual 里 PaintUniformColor(JOINT_COLORS[i])）
            auto smpl_with_joints = std::make_shared<geometry::TriangleMesh>();
            *smpl_with_joints += *smpl_body;
            *smpl_with_joints += *joints_mesh_ptr;

            smpl_with_joints->ComputeTriangleNormals();
            smpl_with_joints->ComputeVertexNormals();

            std::string OUTPUT_SMPL_JOINTS = OUTPUT_DIR + "smpl_fit_with_joints.glb";
            io::WriteTriangleMesh(OUTPUT_SMPL_JOINTS, *smpl_with_joints);
            std::cout << ">>> 已保存: " << OUTPUT_SMPL_JOINTS << " (SMPL + 彩色关节)" << std::endl;
        }


        // 6. [核心] 切换到原始模型进行测量
        //    (务必使用原始扫描，否则切手臂逻辑效果大打折扣)
        std::cout << ">>> 正在切换至原始扫描模型进行测量..." << std::endl;
        auto target_mesh_for_measure = std::make_shared<geometry::TriangleMesh>(*target_ptr);
        measure_14_items(target_mesh_for_measure, final_joints);

    }
    catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}