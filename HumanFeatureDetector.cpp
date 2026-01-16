#include "HumanFeatureDetector.h"

#include <open3d/geometry/PointCloud.h>
#include <open3d/io/TriangleMeshIO.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <map>

using namespace open3d;

// ------------------------ utils ------------------------
std::array<Eigen::Vector3d, 24> HumanFeatureDetector::TensorToJoints24(const torch::Tensor& joints_tensor) {
    torch::Tensor tmp = joints_tensor.squeeze(0).cpu();
    auto acc = tmp.accessor<float, 2>();
    std::array<Eigen::Vector3d, 24> J{};
    for (int i = 0; i < 24; ++i) J[i] = Eigen::Vector3d(acc[i][0], acc[i][1], acc[i][2]);
    return J;
}

std::vector<Eigen::Vector3d> HumanFeatureDetector::GetSliceY(const geometry::TriangleMesh& mesh, double y_height, double tolerance) {
    std::vector<Eigen::Vector3d> slice_points;
    const double min_y = y_height - tolerance / 2.0;
    const double max_y = y_height + tolerance / 2.0;
    slice_points.reserve(mesh.vertices_.size() / 20);
    for (const auto& v : mesh.vertices_) {
        if (v.y() >= min_y && v.y() <= max_y) slice_points.push_back(v);
    }
    return slice_points;
}

std::vector<Eigen::Vector3d> HumanFeatureDetector::GetSliceX(const geometry::TriangleMesh& mesh, double x_val, double tolerance) {
    std::vector<Eigen::Vector3d> slice_points;
    const double min_x = x_val - tolerance / 2.0;
    const double max_x = x_val + tolerance / 2.0;
    slice_points.reserve(mesh.vertices_.size() / 20);
    for (const auto& v : mesh.vertices_) {
        if (v.x() >= min_x && v.x() <= max_x) slice_points.push_back(v);
    }
    return slice_points;
}

// Your integer-bin thickness filter + DBSCAN largest cluster (ported from main.cpp)
std::vector<Eigen::Vector3d> HumanFeatureDetector::GetTorsoSlice(
    const std::vector<Eigen::Vector3d>& full_slice_pts,
    double eps,
    int min_points,
    double thickness_ratio,
    bool debug) {

    if ((int)full_slice_pts.size() < min_points * 2) return full_slice_pts;

    // DBSCAN
    auto pcd = std::make_shared<geometry::PointCloud>();
    pcd->points_ = full_slice_pts;
    std::vector<int> labels = pcd->ClusterDBSCAN(eps, min_points, false);

    std::map<int, int> counts;
    for (int lb : labels) if (lb != -1) counts[lb]++;

    if (debug) {
        std::cout << "\n[DEBUG] DBSCAN Clusters: " << counts.size() << " (eps=" << eps << ")" << std::endl;
        std::vector<std::pair<int, int>> items(counts.begin(), counts.end());
        std::sort(items.begin(), items.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (auto& it : items) std::cout << "  - label " << it.first << " : " << it.second << " pts" << std::endl;
    }

    if (counts.empty()) return full_slice_pts;

    // take largest cluster first
    int max_label = -1, max_c = -1;
    for (auto const& kv : counts) {
        if (kv.second > max_c) { max_c = kv.second; max_label = kv.first; }
    }

    std::vector<Eigen::Vector3d> torso;
    torso.reserve((size_t)max_c);
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == max_label) torso.push_back(full_slice_pts[i]);
    }

    if (debug && counts.size() > 1) std::cout << "  -> 聚类成功分离，取最大簇。" << std::endl;

    // thickness gating along X (same idea as your v27)
    if (torso.size() < (size_t)min_points) return full_slice_pts;

    double min_x = DBL_MAX, max_x = -DBL_MAX;
    for (auto& p : torso) { min_x = std::min(min_x, p.x()); max_x = std::max(max_x, p.x()); }
    if (max_x - min_x < 1e-9) return torso;

    const double step = 0.005; // 5mm bins (in meters)
    const int num_bins = std::max(1, (int)std::ceil((max_x - min_x) / step));

    std::vector<double> min_z(num_bins, DBL_MAX), max_z(num_bins, -DBL_MAX);
    std::vector<char> has(num_bins, 0);

    for (auto& p : torso) {
        int idx = (int)std::floor((p.x() - min_x) / step);
        idx = std::max(0, std::min(num_bins - 1, idx));
        has[idx] = 1;
        min_z[idx] = std::min(min_z[idx], p.z());
        max_z[idx] = std::max(max_z[idx], p.z());
    }

    std::vector<double> thickness(num_bins, 0.0);
    double max_thick = 0.0;
    int peak_idx = 0;
    for (int i = 0; i < num_bins; ++i) {
        if (!has[i]) continue;
        thickness[i] = max_z[i] - min_z[i];
        if (thickness[i] > max_thick) { max_thick = thickness[i]; peak_idx = i; }
    }

    const double depth_threshold = max_thick * thickness_ratio;

    int min_valid_idx = 0;
    int max_valid_idx = num_bins - 1;

    for (int i = peak_idx; i >= 0; --i) {
        if (has[i] && thickness[i] < depth_threshold) { min_valid_idx = i; break; }
    }
    for (int i = peak_idx; i < num_bins; ++i) {
        if (has[i] && thickness[i] < depth_threshold) { max_valid_idx = i; break; }
    }

    std::vector<Eigen::Vector3d> result;
    result.reserve(torso.size());
    for (auto& p : torso) {
        int idx = (int)std::floor((p.x() - min_x) / step);
        if (idx > min_valid_idx && idx < max_valid_idx) result.push_back(p);
    }

    if (result.size() < (size_t)min_points) return torso;
    return result;
}

// 2D convex hull on XZ (Andrew)
static double cross2(const Eigen::Vector2d& O, const Eigen::Vector2d& A, const Eigen::Vector2d& B) {
    return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
}

std::vector<Eigen::Vector3d> HumanFeatureDetector::GetConvexHullPointsXZ(const std::vector<Eigen::Vector3d>& points) {
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
    hull.reserve(H.size());
    for (auto h : H) hull.push_back(points[pts[h].idx]);
    return hull;
}

std::vector<Eigen::Vector3d> HumanFeatureDetector::ResampleClosedRing(const std::vector<Eigen::Vector3d>& ring, int N) {
    std::vector<Eigen::Vector3d> out;
    if (ring.size() < 2 || N < 2) return out;

    std::vector<double> acc(ring.size() + 1, 0.0);
    for (size_t i = 0; i < ring.size(); ++i)
        acc[i + 1] = acc[i] + (ring[(i + 1) % ring.size()] - ring[i]).norm();
    double L = acc.back();
    if (L < 1e-9) return out;

    out.reserve((size_t)N);
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

double HumanFeatureDetector::PerimeterClosed(const std::vector<Eigen::Vector3d>& ring) {
    if (ring.size() < 2) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < ring.size(); ++i)
        sum += (ring[(i + 1) % ring.size()] - ring[i]).norm();
    return sum;
}

std::shared_ptr<geometry::TriangleMesh> HumanFeatureDetector::VisualizeSlicePoints(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Vector3d& color,
    double radius) {

    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (points.empty()) return mesh;

    auto base_sphere = geometry::TriangleMesh::CreateSphere(radius, 6);
    base_sphere->PaintUniformColor(color);

    for (const auto& p : points) {
        auto s = std::make_shared<geometry::TriangleMesh>(*base_sphere);
        s->Translate(p);
        *mesh += *s;
    }
    return mesh;
}

std::shared_ptr<geometry::TriangleMesh> HumanFeatureDetector::VisualizePathTube(
    const std::vector<Eigen::Vector3d>& path,
    const Eigen::Vector3d& color,
    double r) {

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

double HumanFeatureDetector::DistPointSegment(const Eigen::Vector3d& p, const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    Eigen::Vector3d ab = b - a;
    double t = (p - a).dot(ab) / (ab.dot(ab) + 1e-12);
    t = std::max(0.0, std::min(1.0, t));
    Eigen::Vector3d proj = a + t * ab;
    return (p - proj).norm();
}

// ------------------------ per-measurement blocks ------------------------

// 1) neck
double HumanFeatureDetector::MeasureNeck(Context& ctx) {
    double neck_y = ctx.J[12].y() + 0.05;
    auto neck_raw = GetSliceY(*ctx.mesh, neck_y, 0.015);
    auto neck_clean = GetTorsoSlice(neck_raw, 0.02, 5, 0.50, false);

    if (!neck_raw.empty()) *ctx.visuals += *VisualizeSlicePoints(neck_raw, { 1,1,0 }, ctx.vis.vis_sphere_r);

    auto neck_hull = GetConvexHullPointsXZ(neck_clean);
    auto neck_ring = ResampleClosedRing(neck_hull, 120);
    return PerimeterClosed(neck_ring);
}

// 2) chest
double HumanFeatureDetector::MeasureChest(Context& ctx) {
    double chest_y = ctx.J[6].y() * 0.15 + ctx.J[9].y() * 0.85;
    auto chest_raw = GetSliceY(*ctx.mesh, chest_y, 0.02);

    // fixed params you validated
    bool debug = true;
    auto chest_clean = GetTorsoSlice(chest_raw, 0.008, 20, 0.70, debug);

    if (!chest_clean.empty())
        *ctx.visuals += *VisualizeSlicePoints(chest_clean, { 0,1,0 }, ctx.vis.vis_sphere_r);

    if (chest_clean.empty()) chest_clean = chest_raw;

    auto chest_hull = GetConvexHullPointsXZ(chest_clean);
    auto chest_ring = ResampleClosedRing(chest_hull, 240);

    std::cout << "[chest] raw=" << chest_raw.size()
              << " clean=" << chest_clean.size()
              << " hull=" << chest_hull.size()
              << " ring=" << chest_ring.size() << "\n";

    return PerimeterClosed(chest_ring);
}

// 3) waist
double HumanFeatureDetector::MeasureWaist(Context& ctx) {
    ctx.waist_y = ctx.J[3].y();

    auto waist_raw = GetSliceY(*ctx.mesh, ctx.waist_y, 0.02);
    auto waist_clean = GetTorsoSlice(waist_raw, 0.02, 10, 0.55, false);

    auto waist_vis_raw = GetSliceY(*ctx.mesh, ctx.waist_y, ctx.vis.vis_tol_y);
    auto waist_vis_torso = GetTorsoSlice(waist_vis_raw, 0.02, 10, 0.55, false);
    if (!waist_vis_torso.empty())
        *ctx.visuals += *VisualizeSlicePoints(waist_vis_torso, { 1,0,1 }, ctx.vis.vis_sphere_r);

    auto waist_hull = GetConvexHullPointsXZ(waist_clean);
    auto waist_ring = ResampleClosedRing(waist_hull, 200);
    ctx.waist_girth = PerimeterClosed(waist_ring);

    // for pants cylinder placement
    ctx.max_waist_x = -1e9;
    if (!waist_ring.empty()) {
        for (auto& p : waist_ring) ctx.max_waist_x = std::max(ctx.max_waist_x, p.x());
    } else {
        ctx.max_waist_x = ctx.J[0].x() + 0.15;
    }

    return ctx.waist_girth;
}

// 4) hip
double HumanFeatureDetector::MeasureHip(Context& ctx) {
    double hip_best = 0.0;
    double hip_best_y = ctx.J[0].y();

    for (double y = ctx.J[0].y() - 0.12; y < ctx.J[0].y() + 0.04; y += 0.01) {
        auto raw = GetSliceY(*ctx.mesh, y, 0.003);
        if (raw.empty()) continue;
        auto clean = GetTorsoSlice(raw, 0.015, 10, 0.55, false);
        if (clean.size() < 20) continue;
        auto hull = GetConvexHullPointsXZ(clean);
        auto ring = ResampleClosedRing(hull, 240);
        double per = PerimeterClosed(ring);
        if (per > 0.60 && per < 1.30 && per > hip_best) {
            hip_best = per;
            hip_best_y = y;
        }
    }

    auto hip_vis_raw = GetSliceY(*ctx.mesh, hip_best_y, ctx.vis.vis_tol_y);
    auto hip_vis_torso = GetTorsoSlice(hip_vis_raw, 0.02, 10, 0.65, false);
    if (!hip_vis_torso.empty())
        *ctx.visuals += *VisualizeSlicePoints(hip_vis_torso, { 1,0.5,0 }, ctx.vis.vis_sphere_r);

    return hip_best;
}

// 5) thigh (left)
double HumanFeatureDetector::MeasureThigh(Context& ctx) {
    const int L_HIP = 1, R_HIP = 2, L_KNEE = 4;
    double thigh_y0 = 0.60 * ctx.J[L_HIP].y() + 0.40 * ctx.J[L_KNEE].y();

    double thigh_best = 0.0;
    double thigh_best_y = thigh_y0;

    for (double y = thigh_y0 + 0.04; y >= thigh_y0 - 0.06; y -= 0.01) {
        auto raw = GetSliceY(*ctx.mesh, y, 0.02);
        if (raw.empty()) continue;

        std::vector<Eigen::Vector3d> left_leg;
        left_leg.reserve(raw.size());
        for (auto& p : raw) {
            if ((p - ctx.J[L_HIP]).norm() < (p - ctx.J[R_HIP]).norm() - 0.005)
                left_leg.push_back(p);
        }

        auto clean = GetTorsoSlice(left_leg, 0.015, 10, 0.60, false);
        if (clean.size() < 20) continue;

        auto hull = GetConvexHullPointsXZ(clean);
        auto ring = ResampleClosedRing(hull, 200);
        double per = PerimeterClosed(ring);
        if (per > 0.30 && per < 0.90 && per > thigh_best) {
            thigh_best = per;
            thigh_best_y = y;
        }
    }

    // visualization: raw left-leg slice -> DBSCAN -> largest cluster -> points + ring tube
    const double VIS_THIGH_TOL = 0.01;
    const double EPS = 0.015;
    const int MINP = 10;

    auto vis_raw = GetSliceY(*ctx.mesh, thigh_best_y, VIS_THIGH_TOL);
    std::vector<Eigen::Vector3d> vis_left;
    vis_left.reserve(vis_raw.size());
    for (auto& p : vis_raw) {
        if ((p - ctx.J[L_HIP]).norm() < (p - ctx.J[R_HIP]).norm() - 0.005)
            vis_left.push_back(p);
    }

    std::vector<Eigen::Vector3d> vis_big;
    if (!vis_left.empty()) {
        auto pcd = std::make_shared<geometry::PointCloud>();
        pcd->points_ = vis_left;
        auto labels = pcd->ClusterDBSCAN(EPS, MINP, false);

        std::map<int, int> cnt;
        for (int lb : labels) if (lb != -1) cnt[lb]++;

        int best_label = -1, best_sz = 0;
        for (auto& kv : cnt) {
            if (kv.second > best_sz) { best_sz = kv.second; best_label = kv.first; }
        }

        if (best_label != -1) {
            vis_big.reserve((size_t)best_sz);
            for (size_t i = 0; i < vis_left.size(); ++i)
                if (labels[i] == best_label) vis_big.push_back(vis_left[i]);
        }

        std::cout << "[thigh_vis] left=" << vis_left.size()
                  << " clusters=" << cnt.size()
                  << " best=" << best_sz
                  << " (eps=" << EPS << ",minPts=" << MINP << ")\n";
    }

    if (!vis_big.empty())
        *ctx.visuals += *VisualizeSlicePoints(vis_big, { 1,0,0 }, ctx.vis.vis_sphere_r);

    if (!vis_big.empty()) {
        auto hull = GetConvexHullPointsXZ(vis_big);
        auto ring = ResampleClosedRing(hull, 200);
        if (ring.size() >= 3) {
            for (auto& p : ring) p.y() = thigh_best_y;
            *ctx.visuals += *VisualizePathTube(ring, { 1,0,0 }, ctx.vis.tube_r);
        }
    }

    return thigh_best;
}

// 6) shoulder width
static double dist_v3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return (a - b).norm(); }

double HumanFeatureDetector::MeasureShoulderWidth(Context& ctx) {
    double sh_dist = dist_v3(ctx.J[16], ctx.J[17]);
    return sh_dist + 0.12; // keep your bias
}

// 7) armhole (legacy x-slice)
double HumanFeatureDetector::MeasureArmhole(Context& ctx) {
    double x_cut = ctx.J[16].x() + 0.02;
    auto arm_raw = GetSliceX(*ctx.mesh, x_cut, 0.03);

    std::vector<Eigen::Vector3d> arm_clean;
    arm_clean.reserve(arm_raw.size());
    for (auto& p : arm_raw) if ((p - ctx.J[16]).norm() < 0.25) arm_clean.push_back(p);

    if (!arm_clean.empty())
        *ctx.visuals += *VisualizeSlicePoints(arm_clean, { 0,0,1 }, ctx.vis.vis_sphere_r);

    std::vector<Eigen::Vector3d> points_mapped;
    points_mapped.reserve(arm_clean.size());
    for (auto& p : arm_clean) points_mapped.push_back({ p.y(), 0, p.z() });

    auto hull = GetConvexHullPointsXZ(points_mapped);
    std::vector<Eigen::Vector3d> ring;
    ring.reserve(hull.size());
    for (auto& p : hull) ring.push_back({ x_cut, p.x(), p.z() });

    ring = ResampleClosedRing(ring, 120);
    return PerimeterClosed(ring);
}

void HumanFeatureDetector::MeasureRest(Context& ctx, Report& rep) {
    double arm_len = dist_v3(ctx.J[16], ctx.J[18]) + dist_v3(ctx.J[18], ctx.J[20]);

    rep.arm_len_cm = (arm_len + 0.05) * 100.0;
    rep.middle_finger_cm = ctx.height * 0.11 * 0.45 * 100.0;
    rep.hand_len_cm = ctx.height * 0.11 * 100.0;
    rep.hand_width_cm = ctx.height * 0.11 * 0.48 * 100.0;
    rep.torso_vertical_girth_cm = ((ctx.J[12].y() - ctx.J[0].y()) * 2.0 + 0.15) * 100.0;
    rep.coat_len_cm = (ctx.J[12].y() - ctx.J[1].y()) * 100.0;

    double pants_len = (ctx.waist_girth > 0) ? std::max(0.0, (ctx.waist_y - ctx.min_y) - 0.03) : 0.0;
    rep.pants_len_cm = pants_len * 100.0;

    if (pants_len > 0) {
        auto cyl = geometry::TriangleMesh::CreateCylinder(0.008, pants_len, 12);
        cyl->PaintUniformColor({ 0.5, 0, 0.5 });
        cyl->Translate({ ctx.max_waist_x + 0.05, ctx.waist_y - pants_len / 2.0, 0 });
        Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
        cyl->Rotate(R, { ctx.max_waist_x + 0.05, ctx.waist_y - pants_len / 2.0, 0 });
        *ctx.visuals += *cyl;
    }
}

HumanFeatureDetector::Report HumanFeatureDetector::Measure14Items(
    const std::shared_ptr<geometry::TriangleMesh>& mesh,
    const torch::Tensor& joints_tensor,
    const std::string& output_merged_path) {

    Context ctx;
    ctx.mesh = mesh;
    ctx.vis = vis_;
    ctx.visuals = std::make_shared<geometry::TriangleMesh>();
    ctx.J = TensorToJoints24(joints_tensor);

    auto bbox = mesh->GetAxisAlignedBoundingBox();
    ctx.min_y = bbox.GetMinBound().y();
    ctx.height = bbox.GetMaxBound().y() - ctx.min_y;

    Report rep;

    std::cout << "\n================ [ 14项专业身体测量报告 ] ================" << std::endl;
    rep.height_cm = ctx.height * 100.0;
    std::cout << "身高            : " << rep.height_cm << " cm" << std::endl;

    rep.neck_cm  = MeasureNeck(ctx) * 100.0;
    std::cout << "1)  颈围        : " << rep.neck_cm << " cm" << std::endl;

    rep.chest_cm = MeasureChest(ctx) * 100.0;
    std::cout << "2)  胸围        : " << rep.chest_cm << " cm" << std::endl;

    rep.waist_cm = MeasureWaist(ctx) * 100.0;
    std::cout << "3)  腰围        : " << rep.waist_cm << " cm" << std::endl;

    rep.hip_cm   = MeasureHip(ctx) * 100.0;
    std::cout << "4)  臀围        : " << rep.hip_cm << " cm" << std::endl;

    rep.thigh_cm = MeasureThigh(ctx) * 100.0;
    std::cout << "5)  大腿围      : " << rep.thigh_cm << " cm" << std::endl;

    rep.shoulder_width_cm = MeasureShoulderWidth(ctx) * 100.0;
    std::cout << "6)  肩宽        : " << rep.shoulder_width_cm << " cm" << std::endl;

    rep.armhole_cm = MeasureArmhole(ctx) * 100.0;
    std::cout << "7)  袖笼围      : " << rep.armhole_cm << " cm" << std::endl;

    MeasureRest(ctx, rep);
    std::cout << "8)  臂长        : " << rep.arm_len_cm << " cm" << std::endl;
    std::cout << "9)  中指长      : " << rep.middle_finger_cm << " cm (AI)" << std::endl;
    std::cout << "10) 手长        : " << rep.hand_len_cm << " cm (AI)" << std::endl;
    std::cout << "11) 手宽        : " << rep.hand_width_cm << " cm (AI)" << std::endl;
    std::cout << "12) 躯干垂直围  : " << rep.torso_vertical_girth_cm << " cm" << std::endl;
    std::cout << "13) 衣长        : " << rep.coat_len_cm << " cm" << std::endl;
    std::cout << "14) 裤长        : " << rep.pants_len_cm << " cm" << std::endl;

    // append visuals and write
    *mesh += *ctx.visuals;
    io::WriteTriangleMesh(output_merged_path, *mesh);
    std::cout << "✅ 测量结束，模型(含点云)已保存: " << output_merged_path << std::endl;

    return rep;
}
