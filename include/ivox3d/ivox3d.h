#ifndef FASTER_LIO_IVOX3D_H
#define FASTER_LIO_IVOX3D_H

#include <execution>
#include <list>
#include <thread>
#include <unordered_map>
#include <file_logger.h>

#include "eigen_types.h"
#include "ivox3d_node.hpp"

namespace faster_lio {

enum class IVoxNodeType {
    DEFAULT,  // linear ivox
};

/// traits for NodeType
// template <IVoxNodeType node_type, typename PointT, int dim>
// struct IVoxNodeTypeTraits {};

// template <typename PointT, int dim>
// struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim> {
//     using NodeType = IVoxNode<PointT, dim>;
// };

template <int dim = 3, IVoxNodeType node_type = IVoxNodeType::DEFAULT, typename PointType = pcl::PointXYZ>
class IVox {
public:
    using KeyType = Eigen::Matrix<int, dim, 1>;
    using PtType = Eigen::Matrix<float, dim, 1>;
    using NodeType = IVoxNode<PointType, dim>;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using DistPoint = typename NodeType::DistPoint;

    enum class NearbyType {
        CENTER,  // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
        RANDOM,  // 0.3m
    };

    struct Options {
        float resolution_ = 0.2;                         // ivox resolution
        float inv_resolution_ = 10.0;                    // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY18;  // nearby range
        std::size_t capacity_ = 1000000;                 // capacity
        int P_max = 500;
        int P_threshold = 50;
        int add_value = 5;
        int sub_value = 1;
    };

    /**
     * constructor
     * @param options  ivox options
     */
    explicit IVox(Options options) : options_(options) {
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();
    }

    // 把 points_to_add 保存下来。体素数量有上限，一个体素只维护一个最新的体素（1）。
    // 0.2 voxel size; 0.05 resolution. -> 64 points/voxel. -> 设置上限为 50
    void AddPoints(const PointVector& points_to_add, int probability = 0);

    void updateP(std::unordered_set<KeyType, hash_vec<3>>& pass, 
        std::unordered_set<KeyType, hash_vec<3>>& hit);

    int getP(const Eigen::Vector3f& v);

    // get nn with condition
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 0.4);

    // get nn in cloud
    // 暂时不用
    bool GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud);

    // /// get number of points
    // size_t NumPoints() const;

    // get number of valid grids
    size_t NumValidGrids() const;

    // get statistics of the points
    // 暂时不用
    //std::vector<float> StatGridPoints() const;

private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    /// position to grid
    KeyType Pos2Grid(const PtType& pt) const;

    Options options_;
    // <key, value, hash function>
    // 维护一个指向 grids_cache_ 的哈希容器。容器的内存有限，最多 options_.capacity_ 个。
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec<dim>>
        grids_map_;
    // 体素的真正信息存储在这里
    std::list<std::pair<KeyType, NodeType>> grids_cache_;
    std::vector<KeyType> nearby_grids_;  // nearbys
};

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num,
                                                      double max_range) {
    std::vector<DistPoint> candidates;
    candidates.reserve(max_num*max_num*max_num);  // 5*5*5

    auto key = Pos2Grid(ToEigen<float, dim>(pt));

    int radius = 1;
    int dia_3 = radius*2 + 1;
    // skip [0,0,0]
    for (int i=0; i<nearby_grids_.size(); i++) {
        // nearby_grids_ is sorted from near to far.
        // if current candidate is large enough, stop search.
        if (i==dia_3) {
            if (candidates.size()>2*max_num) {
                break;
            }
            else {
                radius++;
                dia_3 = radius*2 + 1;
            }
        }
        const KeyType& delta = nearby_grids_[i];
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            iter->second->second.KNNPointByCondition(candidates, pt, max_num);
        }
    }

    // 排序
    if (candidates.size() <= max_num) {
        return false;
    }
    else {
        // 筛选前 5 近邻
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        candidates.resize(max_num);
    }
    // 最近邻放在首位
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

    closest_pt.clear();
    for (auto& it : candidates) {
        closest_pt.emplace_back(it.Get());
    }
    return closest_pt.empty() == false;
}

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumValidGrids() const {
    return grids_map_.size();
}

template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0),  KeyType(0, 0, -1),  KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0),  KeyType(1, -1, 0),  KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1),  KeyType(0, 1, -1),  KeyType(0, -1, -1), KeyType(1, 1, 1),
                         KeyType(-1, 1, 1),  KeyType(1, -1, 1),  KeyType(1, 1, -1),  KeyType(-1, -1, 1),
                         KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::RANDOM) {
        // nn points radius = 0.3 m
        int max_r = static_cast<int>(0.3 * options_.inv_resolution_);
        for (int radius=0; radius<=max_r; radius++) {
            for (int i=-radius; i<=radius; i++) {
                for (int j=-radius; j<=radius; j++) {
                    for (int k=-radius; k<=radius; k++) {
                        if (abs(i)==radius || abs(j)==radius || abs(k)==radius) {
                            nearby_grids_.emplace_back(KeyType(i,j,k));
                        }
                    }
                }
            }
        }
    } else {
        //LOG(ERROR) << "Unknown nearby_type!";
    }
}

// 暂时不用
template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) {
    std::vector<size_t> index(cloud.size());
    for (int i = 0; i < cloud.size(); ++i) {
        index[i] = i;
    }
    closest_cloud.resize(cloud.size());

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&cloud, &closest_cloud, this](size_t idx) {
        PointType pt;
        if (GetClosestPoint(cloud[idx], pt)) {
            closest_cloud[idx] = pt;
        } else {
            closest_cloud[idx] = PointType();
        }
    });
    return true;
}

template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::AddPoints(const PointVector& points_to_add, int probability) {
    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [&probability, this](const auto& pt) {
        // 转换为体素坐标
        auto key = Pos2Grid(ToEigen<float, dim>(pt));
        // 返回 value 是一个迭代器
        auto iter = grids_map_.find(key);
        // if new
        if (iter == grids_map_.end()) {
            // 保存新的体素信息
            grids_cache_.push_front({key, NodeType(pt, probability)});
            grids_map_.insert({key, grids_cache_.begin()});
            grids_cache_.front().second.InsertPoint(pt);
            // 只能维护一定数量的体素
            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
                neal::logger(neal::LOG_WARN, "grids map runs out of cache.");
            }
        }
        else {
            iter->second->second.InsertPoint(pt);
            // 将 iter->second 剪接到 grids_cache_ 的 begin 位置。
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            grids_map_[key] = grids_cache_.begin();
        }
    });
}

template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::updateP(std::unordered_set<KeyType, hash_vec<3>>& pass, 
    std::unordered_set<KeyType, hash_vec<3>>& hit) {

    for(typename std::unordered_set<KeyType, hash_vec<3>>::iterator it = pass.begin(); it != pass.end(); it++) {
        // 返回 value 是一个迭代器
        auto iter = grids_map_.find(*it);
        // if not new
        if (iter != grids_map_.end()) {
            // 保存新的体素信息
            iter->second->second.subP(options_.sub_value);
        }
    }
    for(typename std::unordered_set<KeyType, hash_vec<3>>::iterator it = hit.begin(); it != hit.end(); it++) {
        // 返回 value 是一个迭代器
        auto iter = grids_map_.find(*it);
        // if not new
        if (iter != grids_map_.end()) {
            // 保存新的体素信息
            iter->second->second.addP(options_.P_max, options_.add_value);
        }
    }
}

template <int dim, IVoxNodeType node_type, typename PointType>
int IVox<dim, node_type, PointType>::getP(const Eigen::Vector3f& v) {
    // 转换为体素坐标
    auto key = Pos2Grid(v);
    auto iter = grids_map_.find(key);
    if (iter == grids_map_.end()) {
        return 0;
    }
    return iter->second->second.getP();
}

template <int dim, IVoxNodeType node_type, typename PointType>
Eigen::Matrix<int, dim, 1> IVox<dim, node_type, PointType>::Pos2Grid(const IVox::PtType& pt) const {
    return (pt * options_.inv_resolution_).array().round().template cast<int>();
}

// template <int dim, IVoxNodeType node_type, typename PointType>
// std::vector<float> IVox<dim, node_type, PointType>::StatGridPoints() const {
//     int num = grids_cache_.size(), valid_num = 0, max = 0, min = 100000000;
//     int sum = 0, sum_square = 0;
//     for (auto& it : grids_cache_) {
//         int s = it.second.Size();
//         valid_num += s > 0;
//         max = s > max ? s : max;
//         min = s < min ? s : min;
//         sum += s;
//         sum_square += s * s;
//     }
//     float ave = float(sum) / num;
//     float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;
//     return std::vector<float>{valid_num, ave, max, min, stddev};
// }

}  // namespace faster_lio

#endif
