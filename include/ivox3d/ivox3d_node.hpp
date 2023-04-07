#include <pcl/common/centroid.h>
#include <algorithm>
#include <cmath>
#include <list>
#include <vector>
#include <unordered_map>

#include <file_logger.h>

namespace faster_lio {

// squared distance of two pcl points
template <typename PointT>
inline double distance2(const PointT& pt1, const PointT& pt2) {
    Eigen::Vector3f d = pt1.getVector3fMap() - pt2.getVector3fMap();
    return d.squaredNorm();
}

// convert from pcl point to eigen
template <typename T, int dim, typename PointType>
inline Eigen::Matrix<T, dim, 1> ToEigen(const PointType& pt) {
    return Eigen::Matrix<T, dim, 1>(pt.x, pt.y, pt.z);
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZ>(const pcl::PointXYZ& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZI>(const pcl::PointXYZI& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZINormal>(const pcl::PointXYZINormal& pt) {
    return pt.getVector3fMap();
}

template <typename PointT, int dim = 3>
class IVoxNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct DistPoint;

    IVoxNode(const PointT& c, int P = 0) : center(c), probability(P) {}
    //IVoxNode(const PointT& center, const float& side_length) {}

    void InsertPoint(const PointT& pt);

    // inline bool Empty() const;

    //inline std::size_t Size() const;

    // inline PointT GetPoint(const std::size_t idx) const;

    int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K);

    void addP(int max, int v) {probability = (probability >= max) ? max : (probability+v);}
    void subP(int v) {probability = (probability <= 0) ? 0 : (probability-v);}
    int getP() const {return probability;}

private:
    std::vector<PointT> points_;
    PointT center;
    int probability;
};

template <typename PointT, int dim>
struct IVoxNode<PointT, dim>::DistPoint {
    double dist = 0;
    PointT point;

    DistPoint() = default;
    DistPoint(const double d, const PointT& pointIn) : dist(d), point(pointIn) {}

    PointT Get() { return point; }

    inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }

    inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
};

template <typename PointT, int dim>
void IVoxNode<PointT, dim>::InsertPoint(const PointT& pt) {
    // 策略 1
    // 只维护最新的两个点
    // points_.emplace_back(pt);
    // if (points_.size() > 2) {
    //     points_.pop_front();
    // }

    // 策略 2
    // 只维护最新的一个点
    // while (points_.size()!=0) {
    //     points_.pop_back();
    // }
    // points_.emplace_back(pt);

    // 策略 3
    // 只维护最靠近中心一个点，且从此不再更新
    if (points_.size()==0) {
        points_.emplace_back(pt);
    }
    else if (points_.size()>1) {
        neal::logger(neal::LOG_WARN, "why more than one point???");
        return;
    }
    else {
        double d1 = distance2(pt, center);
        double d2 = distance2(points_[0], center);
        if (d1 < d2) {
            points_.pop_back();
            points_.emplace_back(pt);
        }
    }
    return;
}

// template <typename PointT, int dim>
// bool IVoxNode<PointT, dim>::Empty() const {
//     return points_.empty();
// }

// template <typename PointT, int dim>
// std::size_t IVoxNode<PointT, dim>::Size() const {
//     return points_.size();
// }

template <typename PointT, int dim>
int IVoxNode<PointT, dim>::KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K) {
    // 其它体素内找到的近邻点个数
    std::size_t old_size = dis_points.size();

    for (const auto& pt : points_) {
        double d = distance2(pt, point);
        // if (d < max_range * max_range) {  // < 0.4*0.4
        dis_points.template emplace_back(DistPoint(d, pt));
    }

    // sort by distance
    // if (old_size + K >= dis_points.size()) {
    // } else {
    //     std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1, dis_points.end());
    //     dis_points.resize(old_size + K);
    // }

    return dis_points.size();
}

}  // namespace faster_lio
