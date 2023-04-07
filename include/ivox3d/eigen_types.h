//
// Created by xiang on 2021/7/16.
//

#ifndef FASTER_LIO_EIGEN_TYPES_H
#define FASTER_LIO_EIGEN_TYPES_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

/// alias for eigen
using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;

using Vec2d = Eigen::Vector2d;
using Vec2f = Eigen::Vector2f;
using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Vec5d = Eigen::Matrix<double, 5, 1>;
using Vec5f = Eigen::Matrix<float, 5, 1>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec15d = Eigen::Matrix<double, 15, 15>;

using Mat1d = Eigen::Matrix<double, 1, 1>;
using Mat3d = Eigen::Matrix3d;
using Mat3f = Eigen::Matrix3f;
using Mat4d = Eigen::Matrix4d;
using Mat4f = Eigen::Matrix4f;
using Mat5d = Eigen::Matrix<double, 5, 5>;
using Mat5f = Eigen::Matrix<float, 5, 5>;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using Mat6f = Eigen::Matrix<float, 6, 6>;
using Mat15d = Eigen::Matrix<double, 15, 15>;

using Quatd = Eigen::Quaterniond;
using Quatf = Eigen::Quaternionf;

namespace faster_lio {

/// less of vector
template <int N>
struct less_vec {
    inline bool operator()(const Eigen::Matrix<int, N, 1>& v1, const Eigen::Matrix<int, N, 1>& v2) const;
};

/// hash of vector
template <int N>
struct hash_vec {
    inline size_t operator()(const Eigen::Matrix<int, N, 1>& v) const;
};

template <>
inline bool less_vec<3>::operator()(const Eigen::Matrix<int, 3, 1>& v1, const Eigen::Matrix<int, 3, 1>& v2) const {
    return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]) && (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] < v2[2]);
}

/// vec 3 hash
template <>
inline size_t hash_vec<3>::operator()(const Eigen::Matrix<int, 3, 1>& v) const {
    return size_t(((v[0]) * 73856109325) ^ ((v[1]) * 471945433) ^ ((v[2]) * 83493562791)) % 10000000000;
}

}  // namespace faster_lio

#endif
