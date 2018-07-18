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
 * @brief Create a prior cost function on a 3D orientation variable (quaternion)
 *
 * The Ceres::NormalPrior cost function only supports a single variable. This is a convenience cost function that
 * applies a prior constraint on a 3D orientation.
 *
 * The cost function is of the form:
 *
 *             ||    [  x - b(1)] ||^2
 *   cost(x) = ||A * [  y - b(2)] ||
 *             ||    [  z - b(3)] ||
 *
 * where, the matrix A and the vector b are fixed and (w, x, y, z) are the components of the 3D orientation
 * (quaternion) variable. Note that the cost function does not include the real-valued component of the quaternion,
 * but only its imaginary components.
 * 
 * The functor can also compute the cost of a subset of the equivalent Euler axes (roll, pitch, or yaw), in the event
 * that we are not interested in all the Euler angles represented by the quaternion.
 * 
 * In case the user is interested in implementing a cost function of the form
 *
 *   cost(X) = (X - mu)^T S^{-1} (X - mu)
 *
 * where, mu is a vector and S is a covariance matrix, then, A = S^{-1/2}, i.e the matrix A is the square root
 * information matrix (the inverse of the covariance).
 */
class NormalPriorOrientation3DCostFunctor
{
public:
  using Euler = fuse_variables::Orientation3DStamped::Euler;

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix, most likely the square root information matrix in order (x, y, yaw)
   * @param[in] b The pose measurement or prior in order (w, x, y, z)
   * @param[in] axes - The Euler angle axes for which we want to compute errors. Defaults to all axes.
   */
  NormalPriorOrientation3DCostFunctor(
    const Eigen::MatrixXd& A,
    const Eigen::Vector4d& b,
    const std::vector<Euler> &axes = {Euler::ROLL, Euler::PITCH, Euler::YAW}) :
      A_(A),
      b_(b),
      all_axes_(false),
      update_vector_(3, false)
  {
    // Cache which axes are set to true, and whether all are set to true
    for_each(
      axes.begin(),
      axes.end(),
      [this](const Euler &axis){ update_vector_[size_t(axis) - size_t(Euler::ROLL)] = 1.0; });

    std::vector<Euler> axes_tmp = axes;
    all_axes_ = (axes.size() == 3 && std::unique(axes_tmp.begin(), axes_tmp.end()) == axes_tmp.end());
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

    std::vector<T> residuals_tmp(A_.cols(), T(0));

    // 2. Map the double array to an Eigen matrix (vector)
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > residuals_map(residuals, A_.rows(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > residuals_tmp_map(residuals_tmp.data(), A_.cols(), 1);

    if (all_axes_)
    {
      // 3a. If we're using all the axes, just use the imaginary coefficients as the residual
      residuals_tmp_map(0) = output[1];
      residuals_tmp_map(1) = output[2];
      residuals_tmp_map(2) = output[3];
    }
    else
    {
      // 3b. If we're using some subset of the axes, separate them into Euler angles, zero out the components that we
      // don't want, and then convert the angles back to a quaternion.
      T roll_diff = Orientation3DStamped::getRoll(output[0], output[1], output[2], output[3]);
      T pitch_diff = Orientation3DStamped::getPitch(output[0], output[1], output[2], output[3]);
      T yaw_diff = Orientation3DStamped::getYaw(output[0], output[1], output[2], output[3]);

      const size_t base_ind = size_t(Euler::ROLL);
      residuals_tmp_map(size_t(Euler::ROLL) - base_ind) = roll_diff;
      residuals_tmp_map(size_t(Euler::PITCH) - base_ind) = pitch_diff;
      residuals_tmp_map(size_t(Euler::YAW) - base_ind) = yaw_diff;
    }

    // 4. Scale the residuals by the square root information matrix to account for the measurement uncertainty.
    residuals_map = A_.template cast<T>() * residuals_tmp_map;

    return true;
  }

private:
  Eigen::MatrixXd A_;  //!< The residual weighting matrix, most likely the square root information matrix
  Eigen::Vector4d b_;  //!< The measured 3D orientation (quaternion) value
  bool all_axes_;
  std::vector<double> update_vector_;
};

}  // namespace fuse_constraints

#endif  // FUSE_CONSTRAINTS_NORMAL_PRIOR_POSE_3D_COST_FUNCTOR_H
