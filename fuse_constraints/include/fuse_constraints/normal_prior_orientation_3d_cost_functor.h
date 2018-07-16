/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Locus Robotics
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef FUSE_CONSTRAINTS_NORMAL_PRIOR_POSE_3D_COST_FUNCTOR_H
#define FUSE_CONSTRAINTS_NORMAL_PRIOR_POSE_3D_COST_FUNCTOR_H

#include <fuse_constraints/util.h>
#include <fuse_variables/orientation_3d_stamped.h>

#include <angles/angles.h>
#include <ceres/internal/disable_warnings.h>
#include <ceres/internal/eigen.h>
#include <ceres/rotation.h>
#include <Eigen/Core>


namespace fuse_constraints
{

/**
 * @brief Create a prior cost function on both the position and orientation variables at once.
 *
 * The Ceres::NormalPrior cost function only supports a single variable. This is a convenience cost function that
 * applies a prior constraint on both the position and orientation variables at once.
 *
 * The cost function is of the form:
 *
 *             ||    [  x - b(0)] ||^2
 *   cost(x) = ||A * [  y - b(1)] ||
 *             ||    [yaw - b(2)] ||
 *
 * where, the matrix A and the vector b are fixed and (x, y, yaw) are the components of the position and orientation
 * variables. In case the user is interested in implementing a cost function of the form
 *
 *   cost(X) = (X - mu)^T S^{-1} (X - mu)
 *
 * where, mu is a vector and S is a covariance matrix, then, A = S^{-1/2}, i.e the matrix A is the square root
 * information matrix (the inverse of the covariance).
 */
class NormalPriorOrientation3DCostFunctor
{
public:
  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix, most likely the square root information matrix in order (x, y, yaw)
   * @param[in] b The pose measurement or prior in order (w, x, y, z)
   */
  NormalPriorOrientation3DCostFunctor(const Eigen::Matrix3d& A, const Eigen::Vector4d& b)  :
    A_(A),
    b_(b)
  {
  }

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   */
  template <typename T>
  bool operator()(const T* const orientation, T* residuals) const
  {
    using fuse_variables::Orientation3DStamped;
  
    // 1. Compute the delta quaternion
    T inverse_quaternion[4] =
    {
      orientation[0],
      -orientation[1],
      -orientation[2],
      -orientation[3]
    };

    T observation[4] =
    {
      T(b_(0)),
      T(b_(1)),
      T(b_(2)),
      T(b_(3))
    };

    T output[4];

    ceres::QuaternionProduct(observation, inverse_quaternion, output);

    // Get the relevant components
    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals_map(residuals);
    T roll_diff = Orientation3DStamped::getRoll(output[0], output[1], output[2], output[3]);  // using_roll ? that_value : 0
    T pitch_diff = Orientation3DStamped::getPitch(output[0], output[1], output[2], output[3]);
    T yaw_diff = Orientation3DStamped::getYaw(output[0], output[1], output[2], output[3]);

    T cy = ceres::cos(yaw_diff * 0.5);
    T sy = ceres::sin(yaw_diff * 0.5);
    T cr = ceres::cos(roll_diff * 0.5);
    T sr = ceres::sin(roll_diff * 0.5);
    T cp = ceres::cos(pitch_diff * 0.5);
    T sp = ceres::sin(pitch_diff * 0.5);

    output[0] = cy * cr * cp + sy * sr * sp;
    output[1] = cy * sr * cp - sy * cr * sp;
    output[2] = cy * cr * sp + sy * sr * cp;
    output[3] = sy * cr * cp - cy * sr * sp;

    residuals_map(0) = output[1];
    residuals_map(1) = output[2];
    residuals_map(2) = output[3];

    // Scale the residuals by the square root information matrix to account for
    // the measurement uncertainty.
    residuals_map = A_.template cast<T>() * residuals_map;

    return true;
  }

private:
  Eigen::Matrix3d A_;  //!< The residual weighting matrix, most likely the square root information matrix
  Eigen::Vector4d b_;  //!< The measured 3D orientation (quaternion) value
};

}  // namespace fuse_constraints

#endif  // FUSE_CONSTRAINTS_NORMAL_PRIOR_POSE_3D_COST_FUNCTOR_H
