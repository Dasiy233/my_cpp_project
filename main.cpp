// ============================================================================
// SMPL Fitting + 14 Measurements
// v27.13-measure-only (单文件版 - 已修复编译错误)
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

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Geometry>
#include <tuple>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <map>
#include <random>
#include <cfloat>

using namespace torch::indexing;
using namespace open3d;
namespace fs = std::filesystem;

// ============================================================================
// 【核心配置】
// ============================================================================
const bool FIX_UPSIDE_DOWN = true;
const bool FIX_FRONT_BACK = true;

// 如需强制旋转(绕Y轴)可改这里，默认 0
const float FORCE_ROTATION_ANGLE = 0.0f;

// 路径 (请根据实际情况修改)
const std::string BASE_DIR = "D:/work/C++/my_cpp_project/";
const std::string MODEL_PATH = BASE_DIR + "smpl_female_30.pt";
const std::string SCAN_PATH = BASE_DIR + "input.obj";
const std::string FACES_PATH = BASE_DIR + "smpl_faces.txt";

const std::string OUTPUT_DIR = BASE_DIR + "results/";
const std::string OUTPUT_OBJ = OUTPUT_DIR + "output_smpl_highres.glb";
const std::string OUTPUT_COMPARISON = OUTPUT_DIR + "result_comparison.glb";
const std::string OUTPUT_JOINTS_ONLY = OUTPUT_DIR + "joints_on_input.glb";
const std::string OUTPUT_MERGED = OUTPUT_DIR + "model_with_lines.glb";

// 训练配置
const float INIT_Y_OFFSET = 0.05f;
const int   NUM_BETAS = 30;
const int   TOTAL_ITERS = 650;


const float INIT_ARM_ANGLE = 1.0f; // T-Pose 防粘连初值
// 可视化“珍珠项链”球半径
const double PEARL_RADIUS = 0.003;

// ============================================================================
// 小工具
// ============================================================================
double dist_v3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return (a - b).norm(); }

// 【修复1】保留此处定义，删除后文重复定义
static inline double clampd(double x, double a, double b) { return std::max(a, std::min(b, x)); }

// 兼容不同 Libtorch 版本：index_put_ 有的版本不接受裸 float 标量，这里统一用 0-d Tensor 写入
static inline void SetTensorAt2D(torch::Tensor& t, int row, int col, float v) {
    auto sv = torch::full({}, v, t.options());
    t.index_put_({ row, col }, sv);
}

// Torch：点到线段距离
torch::Tensor point_segment_distance(torch::Tensor p, torch::Tensor a, torch::Tensor b) {
    auto ab = b - a;
    auto denom = torch::sum(ab * ab) + 1e-8;
    auto t = torch::sum((p - a) * ab) / denom;
    t = torch::clamp(t, 0.0, 1.0);
    auto proj = a + t * ab;
    return torch::norm(p - proj);
}

// Chamfer
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

// ============================================================================
// 有序轮廓点：XZ / YZ 极角排序 + 3点平滑
// ============================================================================
std::vector<Eigen::Vector3d> GetOrderedContourPointsXZ(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) return points;

    Eigen::Vector3d center(0, 0, 0);
    for (const auto& p : points) center += p;
    center /= points.size();

    std::vector<Eigen::Vector3d> sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end(), [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return std::atan2(a.z() - center.z(), a.x() - center.x()) <
            std::atan2(b.z() - center.z(), b.x() - center.x());
        });

    return sorted_points;
}

std::vector<Eigen::Vector3d> GetOrderedContourPointsYZ(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) return points;

    Eigen::Vector3d center(0, 0, 0);
    for (const auto& p : points) center += p;
    center /= points.size();

    std::vector<Eigen::Vector3d> sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end(), [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return std::atan2(a.z() - center.z(), a.y() - center.y()) <
            std::atan2(b.z() - center.z(), b.y() - center.y());
        });

    return sorted_points;
}

// 2D 凸包（Monotone chain）
static double cross2(const Eigen::Vector2d& O, const Eigen::Vector2d& A, const Eigen::Vector2d& B) {
    return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
}
std::vector<Eigen::Vector3d> GetConvexHullPointsXZ(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) return points;

    struct P { Eigen::Vector2d p; size_t idx; };
    std::vector<P> pts;
    pts.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        pts.push_back({ Eigen::Vector2d(points[i].x(), points[i].z()), i });

    std::sort(pts.begin(), pts.end(), [](const P& a, const P& b) {
        return a.p.x() < b.p.x() || (a.p.x() == b.p.x() && a.p.y() < b.p.y());
        });

    std::vector<size_t> H;
    H.reserve(pts.size() * 2);

    // lower
    for (size_t i = 0; i < pts.size(); ++i) {
        while (H.size() >= 2) {
            size_t i1 = H[H.size() - 2];
            size_t i2 = H[H.size() - 1];
            if (cross2(pts[i1].p, pts[i2].p, pts[i].p) <= 0) H.pop_back();
            else break;
        }
        H.push_back(i);
    }

    // upper
    size_t lower_size = H.size();
    for (int i = (int)pts.size() - 2; i >= 0; --i) {
        while (H.size() > lower_size) {
            size_t i1 = H[H.size() - 2];
            size_t i2 = H[H.size() - 1];
            if (cross2(pts[i1].p, pts[i2].p, pts[(size_t)i].p) <= 0) H.pop_back();
            else break;
        }
        H.push_back((size_t)i);
    }
    if (H.size() > 1) H.pop_back();

    std::vector<Eigen::Vector3d> hull;
    hull.reserve(H.size());
    for (auto h : H) hull.push_back(points[pts[h].idx]);

    return GetOrderedContourPointsXZ(hull);
}

// ============================================================================
// 可视化 & 曲线平滑
// ============================================================================

static std::vector<Eigen::Vector3d> RemoveNearDuplicates(const std::vector<Eigen::Vector3d>& pts, double eps = 1e-4) {
    if (pts.size() < 3) return pts;
    std::vector<Eigen::Vector3d> out;
    out.reserve(pts.size());
    out.push_back(pts.front());
    for (size_t i = 1; i < pts.size(); ++i) {
        if ((pts[i] - out.back()).norm() > eps) out.push_back(pts[i]);
    }
    if (out.size() >= 2 && (out.front() - out.back()).norm() < eps) out.pop_back();
    return out;
}

static double ClosedPerimeter(const std::vector<Eigen::Vector3d>& pts) {
    if (pts.size() < 2) return 0.0;
    double L = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        L += (pts[(i + 1) % pts.size()] - pts[i]).norm();
    }
    return L;
}

static std::vector<Eigen::Vector3d> ResampleClosedByCount(const std::vector<Eigen::Vector3d>& in_pts, int n) {
    if ((int)in_pts.size() < 2 || n < 3) return in_pts;

    auto pts = RemoveNearDuplicates(in_pts);
    if ((int)pts.size() < 2) return pts;

    std::vector<double> seg_len(pts.size());
    double total = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        double d = (pts[(i + 1) % pts.size()] - pts[i]).norm();
        seg_len[i] = d;
        total += d;
    }
    if (total < 1e-9) return pts;

    const double step = total / (double)n;
    std::vector<Eigen::Vector3d> out;
    out.reserve(n);

    size_t i = 0;
    double dist_on_edge = 0.0;
    Eigen::Vector3d a = pts[0];
    Eigen::Vector3d b = pts[1];

    out.push_back(pts[0]);
    double target = step;

    double walked = 0.0;
    while ((int)out.size() < n) {
        double edge_len = seg_len[i];
        if (edge_len < 1e-12) {
            i = (i + 1) % pts.size();
            a = pts[i];
            b = pts[(i + 1) % pts.size()];
            dist_on_edge = 0.0;
            continue;
        }

        double remain = edge_len - dist_on_edge;
        if (walked + remain >= target) {
            double need = target - walked;
            double t = (dist_on_edge + need) / edge_len;
            t = std::min(1.0, std::max(0.0, t));
            Eigen::Vector3d p = a + (b - a) * t;
            out.push_back(p);
            walked = target;
            target += step;
        }
        else {
            walked += remain;
            i = (i + 1) % pts.size();
            a = pts[i];
            b = pts[(i + 1) % pts.size()];
            dist_on_edge = 0.0;
        }
    }

    if (out.size() >= 2 && (out.front() - out.back()).norm() < 1e-4) out.pop_back();
    return out;
}

static std::vector<Eigen::Vector3d> ChaikinSmoothClosed(const std::vector<Eigen::Vector3d>& in_pts, int iters = 2) {
    if (in_pts.size() < 3) return in_pts;
    std::vector<Eigen::Vector3d> pts = in_pts;

    for (int it = 0; it < iters; ++it) {
        std::vector<Eigen::Vector3d> out;
        out.reserve(pts.size() * 2);

        for (size_t i = 0; i < pts.size(); ++i) {
            const Eigen::Vector3d& p0 = pts[i];
            const Eigen::Vector3d& p1 = pts[(i + 1) % pts.size()];
            out.push_back(p0 * 0.75 + p1 * 0.25);
            out.push_back(p0 * 0.25 + p1 * 0.75);
        }

        pts = RemoveNearDuplicates(out);
        if (pts.size() < 3) break;
    }
    return pts;
}

static std::vector<Eigen::Vector3d> SnapPathToRawXZ(
    const std::vector<Eigen::Vector3d>& path_xz,
    const std::vector<Eigen::Vector3d>& raw_slice)
{
    if (path_xz.size() < 2 || raw_slice.size() < 2) return path_xz;

    auto pcd = std::make_shared<geometry::PointCloud>();
    pcd->points_.reserve(raw_slice.size());
    for (const auto& p : raw_slice) {
        pcd->points_.push_back(Eigen::Vector3d(p.x(), 0.0, p.z()));
    }

    geometry::KDTreeFlann kdtree(*pcd);
    std::vector<Eigen::Vector3d> snapped;
    snapped.reserve(path_xz.size());

    std::vector<int> idx(1);
    std::vector<double> dist2(1);

    for (const auto& p : path_xz) {
        Eigen::Vector3d q(p.x(), 0.0, p.z());
        int k = kdtree.SearchKNN(q, 1, idx, dist2);
        if (k > 0) snapped.push_back(raw_slice[(size_t)idx[0]]);
        else snapped.push_back(p);
    }
    return snapped;
}

static std::vector<Eigen::Vector3d> SnapPathToRawYZ(
    const std::vector<Eigen::Vector3d>& path_yz,
    const std::vector<Eigen::Vector3d>& raw_slice)
{
    if (path_yz.size() < 2 || raw_slice.size() < 2) return path_yz;

    auto pcd = std::make_shared<geometry::PointCloud>();
    pcd->points_.reserve(raw_slice.size());
    for (const auto& p : raw_slice) {
        pcd->points_.push_back(Eigen::Vector3d(0.0, p.y(), p.z()));
    }

    geometry::KDTreeFlann kdtree(*pcd);
    std::vector<Eigen::Vector3d> snapped;
    snapped.reserve(path_yz.size());

    std::vector<int> idx(1);
    std::vector<double> dist2(1);

    for (const auto& p : path_yz) {
        Eigen::Vector3d q(0.0, p.y(), p.z());
        int k = kdtree.SearchKNN(q, 1, idx, dist2);
        if (k > 0) snapped.push_back(raw_slice[(size_t)idx[0]]);
        else snapped.push_back(p);
    }
    return snapped;
}

static std::vector<Eigen::Vector3d> MakeNiceClosedPathXZ(
    const std::vector<Eigen::Vector3d>& ordered_pts,
    const std::vector<Eigen::Vector3d>& raw_slice_pts,
    int resample_n = 220)
{
    if (ordered_pts.size() < 3) return ordered_pts;
    std::vector<Eigen::Vector3d> pts = ordered_pts;
    for (auto& p : pts) p.y() = 0.0;

    pts = RemoveNearDuplicates(pts);
    int n = std::max(120, resample_n);
    pts = ResampleClosedByCount(pts, n);
    pts = ChaikinSmoothClosed(pts, 2);
    pts = RemoveNearDuplicates(pts);
    pts = ResampleClosedByCount(pts, n);

    auto snapped = SnapPathToRawXZ(pts, raw_slice_pts);
    snapped = RemoveNearDuplicates(snapped);
    return snapped;
}

static std::vector<Eigen::Vector3d> MakeNiceClosedPathYZ(
    const std::vector<Eigen::Vector3d>& ordered_pts,
    const std::vector<Eigen::Vector3d>& raw_slice_pts,
    int resample_n = 220)
{
    if (ordered_pts.size() < 3) return ordered_pts;
    std::vector<Eigen::Vector3d> pts = ordered_pts;
    for (auto& p : pts) p.x() = 0.0;

    pts = RemoveNearDuplicates(pts);
    int n = std::max(120, resample_n);
    pts = ResampleClosedByCount(pts, n);
    pts = ChaikinSmoothClosed(pts, 2);
    pts = RemoveNearDuplicates(pts);
    pts = ResampleClosedByCount(pts, n);

    auto snapped = SnapPathToRawYZ(pts, raw_slice_pts);
    snapped = RemoveNearDuplicates(snapped);
    return snapped;
}


// 画“管线”
static std::shared_ptr<geometry::TriangleMesh> VisualizePathTube(
    const std::vector<Eigen::Vector3d>& path_points,
    const Eigen::Vector3d& color,
    double radius = 0.0035,
    int cyl_resolution = 20,
    int sphere_resolution = 10
) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (path_points.size() < 2) return mesh;

    for (const auto& p : path_points) {
        auto sphere = geometry::TriangleMesh::CreateSphere(radius * 0.9, sphere_resolution);
        sphere->Translate(p);
        sphere->PaintUniformColor(color);
        *mesh += *sphere;
    }

    const Eigen::Vector3d z_axis(0, 0, 1);
    for (size_t i = 0; i < path_points.size(); ++i) {
        const Eigen::Vector3d& p1 = path_points[i];
        const Eigen::Vector3d& p2 = path_points[(i + 1) % path_points.size()];
        Eigen::Vector3d v = p2 - p1;
        double len = v.norm();
        if (len < 1e-9) continue;

        auto cyl = geometry::TriangleMesh::CreateCylinder(radius, len, cyl_resolution);
        cyl->PaintUniformColor(color);

        Eigen::Vector3d dir = v / len;
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(z_axis, dir);
        cyl->Rotate(q.toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
        cyl->Translate((p1 + p2) * 0.5);

        *mesh += *cyl;
    }
    return mesh;
}

// 【修复2】已删除旧版 RemoveArmPointsXZ，防止与后文重复

// 兼容旧接口：水平截面
// 【修复3】添加默认参数 = 0.0
static std::shared_ptr<geometry::TriangleMesh> VisualizePath(
    const std::vector<Eigen::Vector3d>& path_points,
    const Eigen::Vector3d& color,
    double /*force_y_height*/ = 0.0)
{
    return VisualizePathTube(path_points, color);
}

// 兼容旧接口：袖笼截面
// 【修复4】添加默认参数 = 0.0
static std::shared_ptr<geometry::TriangleMesh> VisualizePathArmhole(
    const std::vector<Eigen::Vector3d>& path_points,
    const Eigen::Vector3d& color,
    double /*cut_x*/ = 0.0)
{
    return VisualizePathTube(path_points, color);
}


// ============================================================================
// [v27.14] Measurement & Visualization (Dense sampling + Outer-ring projection)
// ============================================================================

static double PerimeterClosed(const std::vector<Eigen::Vector3d>& ring) {
    if (ring.size() < 2) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < ring.size(); ++i) sum += (ring[(i + 1) % ring.size()] - ring[i]).norm();
    return sum;
}

static Eigen::Vector3d NearestPointBruteforce(const std::vector<Eigen::Vector3d>& pts, const Eigen::Vector3d& q) {
    if (pts.empty()) return q;
    double best_d2 = DBL_MAX;
    Eigen::Vector3d best = pts[0];
    for (const auto& p : pts) {
        double d2 = (p - q).squaredNorm();
        if (d2 < best_d2) { best_d2 = d2; best = p; }
    }
    return best;
}

static std::vector<Eigen::Vector3d> ResampleClosedRing(const std::vector<Eigen::Vector3d>& ring, int N) {
    std::vector<Eigen::Vector3d> out;
    if (ring.size() < 2 || N < 2) return out;

    std::vector<double> acc(ring.size() + 1, 0.0);
    for (size_t i = 0; i < ring.size(); ++i) {
        acc[i + 1] = acc[i] + (ring[(i + 1) % ring.size()] - ring[i]).norm();
    }
    double L = acc.back();
    if (L < 1e-9) return out;

    out.reserve(N);
    for (int k = 0; k < N; ++k) {
        double s = (L * k) / (double)N; // [0, L)
        // locate segment
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

static std::vector<Eigen::Vector3d> SmoothClosedRing(const std::vector<Eigen::Vector3d>& ring, int iters = 2, int win = 2) {
    if (ring.size() < 5) return ring;
    std::vector<Eigen::Vector3d> cur = ring;
    std::vector<Eigen::Vector3d> nxt = ring;
    int n = (int)ring.size();
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d sum(0, 0, 0);
            int cnt = 0;
            for (int k = -win; k <= win; ++k) {
                sum += cur[(i + k + n) % n];
                cnt++;
            }
            nxt[i] = sum / (double)cnt;
        }
        cur.swap(nxt);
    }
    return cur;
}

// ------------------ 切片：使用“点云”(从 mesh 采样) ------------------

static std::vector<Eigen::Vector3d> SlicePlaneY_CircleXZ(
    const std::vector<Eigen::Vector3d>& pcd_points,
    double y,
    const Eigen::Vector3d& center,
    double radius_xz,
    double thickness_y)
{
    std::vector<Eigen::Vector3d> slice;
    double r2 = radius_xz * radius_xz;
    for (const auto& p : pcd_points) {
        if (std::abs(p.y() - y) > thickness_y) continue;
        double dx = p.x() - center.x();
        double dz = p.z() - center.z();
        if (dx * dx + dz * dz <= r2) slice.push_back(p);
    }
    return slice;
}

static std::vector<Eigen::Vector3d> SlicePlaneX_CircleYZ(
    const std::vector<Eigen::Vector3d>& pcd_points,
    double x,
    const Eigen::Vector3d& center,
    double radius_yz,
    double thickness_x)
{
    std::vector<Eigen::Vector3d> slice;
    double r2 = radius_yz * radius_yz;
    for (const auto& p : pcd_points) {
        if (std::abs(p.x() - x) > thickness_x) continue;
        double dy = p.y() - center.y();
        double dz = p.z() - center.z();
        if (dy * dy + dz * dz <= r2) slice.push_back(p);
    }
    return slice;
}

// ------------------ 过滤：去掉手臂/手掌附近点 (防止胸/腰/臀被手臂污染) ------------------

static std::vector<Eigen::Vector3d> RemoveArmPointsXZ(
    const std::vector<Eigen::Vector3d>& pts,
    const std::vector<Eigen::Vector3d>& J,
    double cut_r = 0.10)
{
    if (pts.empty()) return pts;
    double r2 = cut_r * cut_r;

    // wrists, hands, elbows
    std::vector<int> arm_ids = { 18,19,20,21,22,23 };
    std::vector<Eigen::Vector2d> arm_xz;
    arm_xz.reserve(arm_ids.size());
    for (int id : arm_ids) arm_xz.emplace_back(J[id].x(), J[id].z());

    std::vector<Eigen::Vector3d> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        Eigen::Vector2d xz(p.x(), p.z());
        bool near_arm = false;
        for (const auto& a : arm_xz) {
            double dx = xz.x() - a.x();
            double dz = xz.y() - a.y();
            if (dx * dx + dz * dz < r2) { near_arm = true; break; }
        }
        if (!near_arm) out.push_back(p);
    }
    return out;
}

// ------------------ 轮廓：角度分桶取“最外点” (XZ / YZ) ------------------

static std::vector<Eigen::Vector3d> OuterRingByAngleBins_XZ(
    const std::vector<Eigen::Vector3d>& slice,
    const Eigen::Vector2d& center_xz,
    int bins)
{
    if ((int)slice.size() < 50) return {};

    std::vector<double> best_r(bins, -1.0);
    std::vector<Eigen::Vector3d> best_p(bins);

    for (const auto& p : slice) {
        double dx = p.x() - center_xz.x();
        double dz = p.z() - center_xz.y();
        double r = std::sqrt(dx * dx + dz * dz);
        if (r < 1e-6) continue;
        double ang = std::atan2(dz, dx); // [-pi,pi]
        int bin = (int)std::floor((ang + M_PI) / (2.0 * M_PI) * bins);
        bin = std::max(0, std::min(bins - 1, bin));
        if (r > best_r[bin]) { best_r[bin] = r; best_p[bin] = p; }
    }

    std::vector<Eigen::Vector3d> ring;
    ring.reserve(bins);
    for (int i = 0; i < bins; ++i) if (best_r[i] > 0.0) ring.push_back(best_p[i]);

    // 重新按角度排序
    std::sort(ring.begin(), ring.end(), [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        double aa = std::atan2(a.z() - center_xz.y(), a.x() - center_xz.x());
        double bb = std::atan2(b.z() - center_xz.y(), b.x() - center_xz.x());
        return aa < bb;
        });

    return ring;
}

static std::vector<Eigen::Vector3d> OuterRingByAngleBins_YZ(
    const std::vector<Eigen::Vector3d>& slice,
    const Eigen::Vector2d& center_yz,
    int bins)
{
    if ((int)slice.size() < 50) return {};

    std::vector<double> best_r(bins, -1.0);
    std::vector<Eigen::Vector3d> best_p(bins);

    for (const auto& p : slice) {
        double dy = p.y() - center_yz.x();
        double dz = p.z() - center_yz.y();
        double r = std::sqrt(dy * dy + dz * dz);
        if (r < 1e-6) continue;
        double ang = std::atan2(dz, dy);
        int bin = (int)std::floor((ang + M_PI) / (2.0 * M_PI) * bins);
        bin = std::max(0, std::min(bins - 1, bin));
        if (r > best_r[bin]) { best_r[bin] = r; best_p[bin] = p; }
    }

    std::vector<Eigen::Vector3d> ring;
    ring.reserve(bins);
    for (int i = 0; i < bins; ++i) if (best_r[i] > 0.0) ring.push_back(best_p[i]);

    std::sort(ring.begin(), ring.end(), [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        double aa = std::atan2(a.z() - center_yz.y(), a.y() - center_yz.x());
        double bb = std::atan2(b.z() - center_yz.y(), b.y() - center_yz.x());
        return aa < bb;
        });

    return ring;
}

// ------------------ 自适应构建闭环 (多 thickness 尝试) ------------------

static std::vector<Eigen::Vector3d> BuildNiceRingXZ_Adaptive(
    const std::vector<Eigen::Vector3d>& pcd_points,
    double y,
    const Eigen::Vector3d& center,
    double radius_xz,
    const std::vector<Eigen::Vector3d>& J, // 用于去手臂点，可传空
    bool remove_arms,
    int bins = 240)
{
    std::vector<double> thickness_try = { 0.006, 0.010, 0.015, 0.020 };
    std::vector<Eigen::Vector3d> best_ring;
    double best_score = -1.0;

    Eigen::Vector2d center_xz(center.x(), center.z());

    for (double th : thickness_try) {
        auto slice = SlicePlaneY_CircleXZ(pcd_points, y, center, radius_xz, th);
        if (remove_arms) slice = RemoveArmPointsXZ(slice, J, 0.11);

        if ((int)slice.size() < 200) continue;

        auto ring = OuterRingByAngleBins_XZ(slice, center_xz, bins);
        if ((int)ring.size() < bins * 0.55) continue; // 覆盖不足会穿模

        // 投影到同一高度，平滑并回吸附到 slice
        for (auto& p : ring) p.y() = y;

        ring = SmoothClosedRing(ring, 2, 2);
        ring = ResampleClosedRing(ring, bins);

        for (auto& p : ring) {
            p.y() = y;
            p = NearestPointBruteforce(slice, p);
            p.y() = y;
        }

        double per = PerimeterClosed(ring);
        double coverage = (double)ring.size() / (double)bins;
        double score = coverage * 1000.0 + per; // coverage 优先

        if (score > best_score) { best_score = score; best_ring = ring; }
    }

    return best_ring;
}

static std::vector<Eigen::Vector3d> BuildNiceRingYZ_Adaptive(
    const std::vector<Eigen::Vector3d>& pcd_points,
    double x_cut,
    const Eigen::Vector3d& center,
    double radius_yz,
    int bins = 240)
{
    std::vector<double> thickness_try = { 0.006, 0.010, 0.015, 0.020 };
    std::vector<Eigen::Vector3d> best_ring;
    double best_score = -1.0;

    Eigen::Vector2d center_yz(center.y(), center.z());

    for (double th : thickness_try) {
        auto slice = SlicePlaneX_CircleYZ(pcd_points, x_cut, center, radius_yz, th);
        if ((int)slice.size() < 200) continue;

        auto ring = OuterRingByAngleBins_YZ(slice, center_yz, bins);
        if ((int)ring.size() < bins * 0.55) continue;

        for (auto& p : ring) p.x() = x_cut;

        ring = SmoothClosedRing(ring, 2, 2);
        ring = ResampleClosedRing(ring, bins);

        for (auto& p : ring) {
            p.x() = x_cut;
            p = NearestPointBruteforce(slice, p);
            p.x() = x_cut;
        }

        double per = PerimeterClosed(ring);
        double coverage = (double)ring.size() / (double)bins;
        double score = coverage * 1000.0 + per;

        if (score > best_score) { best_score = score; best_ring = ring; }
    }

    return best_ring;
}

// ------------------ 腿：强制只保留单侧 ------------------

static std::vector<Eigen::Vector3d> KeepLegSideX(const std::vector<Eigen::Vector3d>& pts, double mid_x, bool want_left) {
    std::vector<Eigen::Vector3d> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        if (want_left) {
            if (p.x() <= mid_x + 0.01) out.push_back(p);
        }
        else {
            if (p.x() >= mid_x - 0.01) out.push_back(p);
        }
    }
    return out;
}

// ================= 核心：14项测量逻辑 (v27.14 稳定画线) =================
void measure_14_items(std::shared_ptr<geometry::TriangleMesh>& mesh, const torch::Tensor& joints_tensor) {
    torch::Tensor joints_cpu = joints_tensor.squeeze(0).cpu();
    auto joints_acc = joints_cpu.accessor<float, 2>();
    std::vector<Eigen::Vector3d> J(24);
    for (int i = 0; i < 24; ++i) J[i] = Eigen::Vector3d(joints_acc[i][0], joints_acc[i][1], joints_acc[i][2]);

    auto bbox = mesh->GetAxisAlignedBoundingBox();
    double min_y = bbox.GetMinBound().y();
    double height = bbox.GetMaxBound().y() - min_y;

    // 关键：对 SMPL Mesh 进行高密度采样用于切片/画线
    const int MEASURE_PCD_N = 160000;
    auto meas_pcd = mesh->SamplePointsUniformly(MEASURE_PCD_N);
    const auto& P = meas_pcd->points_;

    auto all_visuals = std::make_shared<geometry::TriangleMesh>();

    std::cout << "\n================ [ 14项专业身体测量报告 (v27.14 稳定画线版) ] ================" << std::endl;
    std::cout << "身高            : " << height * 100.0 << " cm" << std::endl;

    // 1) 颈围
    double neck_y = J[12].y() + 0.05;
    auto neck_ring = BuildNiceRingXZ_Adaptive(P, neck_y, J[12], 0.20, J, false, 240);
    double neck_girth = PerimeterClosed(neck_ring);
    std::cout << "1)  颈围        : " << neck_girth * 100.0 << " cm" << std::endl;
    if (!neck_ring.empty()) *all_visuals += *VisualizePath(neck_ring, Eigen::Vector3d(1.0, 1.0, 0.0));

    // 2) 胸围
    double chest_y = J[6].y();
    auto chest_ring = BuildNiceRingXZ_Adaptive(P, chest_y, J[6], 0.40, J, true, 240);
    double chest_girth = PerimeterClosed(chest_ring);
    std::cout << "2)  胸围        : " << chest_girth * 100.0 << " cm" << std::endl;
    if (!chest_ring.empty()) *all_visuals += *VisualizePath(chest_ring, Eigen::Vector3d(0.0, 1.0, 0.0));

    // 3) 腰围
    Eigen::Vector3d waist_center = (J[3] + J[0]) * 0.5;
    double waist_best = 1e9;
    std::vector<Eigen::Vector3d> waist_ring_best;
    double waist_y_best = (J[0].y() + J[3].y()) * 0.5;

    double max_waist_x = -DBL_MAX;
    for (double y = J[0].y() + 0.02; y < J[3].y() - 0.02; y += 0.01) {
        auto ring = BuildNiceRingXZ_Adaptive(P, y, waist_center, 0.38, J, true, 200);
        double per = PerimeterClosed(ring);
        if (per <= 0.0) continue;
        if (per < 0.55 || per > 1.40) continue; // 合理范围过滤
        if (per < waist_best) {
            waist_best = per;
            waist_ring_best = ring;
            waist_y_best = y;
            for (auto& pt : ring) max_waist_x = std::max(max_waist_x, pt.x());
        }
    }
    if (waist_best > 1e8) waist_best = 0.0;
    std::cout << "3)  腰围        : " << waist_best * 100.0 << " cm" << std::endl;
    if (!waist_ring_best.empty()) *all_visuals += *VisualizePath(waist_ring_best, Eigen::Vector3d(1.0, 0.0, 1.0));

    // 4) 臀围
    double hip_best = 0.0;
    std::vector<Eigen::Vector3d> hip_ring_best;
    double hip_y_best = J[0].y() - 0.02;

    for (double y = J[0].y() - 0.20; y < J[0].y() - 0.01; y += 0.01) {
        auto ring = BuildNiceRingXZ_Adaptive(P, y, J[0], 0.42, J, true, 220);
        double per = PerimeterClosed(ring);
        if (per <= 0.0) continue;
        if (per < 0.70 || per > 1.80) continue;
        if (per > hip_best) {
            hip_best = per;
            hip_ring_best = ring;
            hip_y_best = y;
        }
    }
    std::cout << "4)  臀围        : " << hip_best * 100.0 << " cm" << std::endl;
    if (!hip_ring_best.empty()) *all_visuals += *VisualizePath(hip_ring_best, Eigen::Vector3d(1.0, 0.5, 0.0));

    // 5) 大腿围
    int L_HIP = 1, R_HIP = 2, L_KNEE = 4;
    bool left_is_negative = (J[L_HIP].x() < J[R_HIP].x());
    double mid_x_lr = 0.5 * (J[L_HIP].x() + J[R_HIP].x());
    double thigh_y0 = 0.60 * J[L_HIP].y() + 0.40 * J[L_KNEE].y();
    double thigh_best = 0.0;
    std::vector<Eigen::Vector3d> thigh_ring_best;
    double thigh_y_best = thigh_y0;

    for (double y = thigh_y0 + 0.04; y >= thigh_y0 - 0.06; y -= 0.01) {
        // 先取切片再单侧过滤，避免两腿混合
        auto slice = SlicePlaneY_CircleXZ(P, y, J[L_HIP], 0.22, 0.012);
        slice = KeepLegSideX(slice, mid_x_lr, left_is_negative); // 只保留左腿
        if ((int)slice.size() < 200) continue;

        Eigen::Vector2d c(J[L_HIP].x(), J[L_HIP].z());
        auto ring = OuterRingByAngleBins_XZ(slice, c, 200);
        if ((int)ring.size() < 120) continue;
        for (auto& p : ring) p.y() = y;
        ring = SmoothClosedRing(ring, 2, 2);
        ring = ResampleClosedRing(ring, 200);
        for (auto& p : ring) { p.y() = y; p = NearestPointBruteforce(slice, p); p.y() = y; }

        double per = PerimeterClosed(ring);
        if (per < 0.35 || per > 1.20) continue;
        if (per > thigh_best) { thigh_best = per; thigh_ring_best = ring; thigh_y_best = y; }
    }

    std::cout << "5)  大腿围      : " << thigh_best * 100.0 << " cm" << std::endl;
    if (!thigh_ring_best.empty()) *all_visuals += *VisualizePath(thigh_ring_best, Eigen::Vector3d(1.0, 0.0, 0.0));

    // 6) 肩宽
    double shoulder_dist = dist_v3(J[16], J[17]);
    std::cout << "6)  肩宽        : " << (shoulder_dist + 0.12) * 100.0 << " cm" << std::endl;
    {
        auto cylinder = geometry::TriangleMesh::CreateCylinder(0.005, shoulder_dist, 12);
        cylinder->PaintUniformColor(Eigen::Vector3d(0, 1, 1));
        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d vec = (J[17] - J[16]);
        double len = vec.norm();
        if (len > 1e-9) {
            vec /= len;
            Eigen::Vector3d axis = z_axis.cross(vec);
            double axis_n = axis.norm();
            double dotv = clampd(z_axis.dot(vec), -1.0, 1.0);
            double angle = std::acos(dotv);
            if (axis_n > 1e-9 && std::abs(angle) > 1e-9) {
                axis /= axis_n;
                Eigen::AngleAxisd rot(angle, axis);
                cylinder->Rotate(rot.toRotationMatrix(), Eigen::Vector3d(0, 0, 0));
            }
            cylinder->Translate((J[16] + J[17]) * 0.5);
            *all_visuals += *cylinder;
        }
    }

    // 7) 袖笼围
    {
        double x_cut = J[16].x() + 0.02;
        auto armhole_ring = BuildNiceRingYZ_Adaptive(P, x_cut, J[16], 0.22, 220);
        double armhole = PerimeterClosed(armhole_ring);
        std::cout << "7)  袖笼围      : " << armhole * 100.0 << " cm" << std::endl;
        if (!armhole_ring.empty()) *all_visuals += *VisualizePath(armhole_ring, Eigen::Vector3d(0.0, 0.0, 1.0));
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

    double pants_len = 0.0;
    if (waist_best > 0.0) pants_len = std::max(0.0, (waist_y_best - min_y) - 0.03);
    std::cout << "14) 裤长        : " << pants_len * 100.0 << " cm" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 裤长可视化
    if (pants_len > 0.0 && max_waist_x > -1e8) {
        Eigen::Vector3d start_p(max_waist_x + 0.05, waist_y_best, 0.0);
        Eigen::Vector3d end_p(start_p.x(), min_y, start_p.z());
        double len = (start_p - end_p).norm();
        auto cylinder = geometry::TriangleMesh::CreateCylinder(0.008, len, 12);
        cylinder->PaintUniformColor({ 0.5, 0.0, 0.5 });
        // 默认沿 Z，转到 Y
        Eigen::Matrix3d R_cyl = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
        cylinder->Rotate(R_cyl, Eigen::Vector3d(0, 0, 0));
        cylinder->Translate((start_p + end_p) * 0.5);
        *all_visuals += *cylinder;
    }

    *mesh += *all_visuals;
    io::WriteTriangleMesh(OUTPUT_MERGED, *mesh);
    std::cout << "✅ 最终合体模型(含测量线)已保存: " << OUTPUT_MERGED << std::endl;
}

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
        if (target_mesh_ptr == nullptr || target_mesh_ptr->vertices_.empty())
            throw std::runtime_error("读取扫描失败");

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

        // 采样点云
        std::cout << ">>> 正在采样点云 (10000 点)..." << std::endl;
        auto pcd = target_mesh_ptr->SamplePointsUniformly(10000);
        auto vertices_eigen = pcd->points_;
        int num_verts = (int)vertices_eigen.size();
        if (num_verts <= 0) throw std::runtime_error("采样失败：点数为0");

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
        torch::NoGradGuard no_grad;

        std::vector<torch::jit::IValue> in;
        in.push_back(betas); in.push_back(pose); in.push_back(global_orient); in.push_back(transl); in.push_back(scale);
        auto outputs = smpl_module.forward(in).toTuple();
        auto final_verts = outputs->elements()[0].toTensor().squeeze(0).cpu();
        auto final_joints = outputs->elements()[1].toTensor();

        auto smpl_mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        auto v_acc = final_verts.accessor<float, 2>();
        for (int k = 0; k < final_verts.size(0); ++k)
            smpl_mesh_ptr->vertices_.push_back(Eigen::Vector3d(v_acc[k][0], v_acc[k][1], v_acc[k][2]));

        if (load_faces(FACES_PATH, smpl_mesh_ptr->triangles_)) {
            smpl_mesh_ptr->ComputeVertexNormals();
            io::WriteTriangleMesh(OUTPUT_OBJ, *smpl_mesh_ptr);
        }
        smpl_mesh_ptr->PaintUniformColor(Eigen::Vector3d(0.8, 0.8, 0.8));

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