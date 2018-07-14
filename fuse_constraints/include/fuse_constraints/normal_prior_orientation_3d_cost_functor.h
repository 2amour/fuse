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

    T roll1 = Orientation3DStamped::getRoll(orientation[0], orientation[1], orientation[2], orientation[3]);
    T pitch1 = Orientation3DStamped::getPitch(orientation[0], orientation[1], orientation[2], orientation[3]);
    T yaw1 = Orientation3DStamped::getYaw(orientation[0], orientation[1], orientation[2], orientation[3]);

    T roll2 = T(Orientation3DStamped::getRoll(b_(0), b_(1), b_(2), b_(3)));
    T pitch2 = T(Orientation3DStamped::getPitch(b_(0), b_(1), b_(2), b_(3)));
    T yaw2 = T(Orientation3DStamped::getYaw(b_(0), b_(1), b_(2), b_(3)));

    // Residual can just be the imaginary components
    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals_map(residuals);
    residuals_map(0) = wrapAngle2D(roll2 - roll1);
    residuals_map(1) = wrapAngle2D(pitch2 - pitch1);
    residuals_map(2) = wrapAngle2D(yaw2 - yaw1);

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
