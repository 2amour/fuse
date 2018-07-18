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
#include <fuse_constraints/absolute_orientation_3d_stamped_constraint.h>
#include <fuse_constraints/normal_prior_orientation_3d_cost_functor.h>

#include <ceres/autodiff_cost_function.h>


namespace fuse_constraints
{

AbsoluteOrientation3DStampedConstraint::AbsoluteOrientation3DStampedConstraint(
  const fuse_variables::Orientation3DStamped& orientation,
  const Eigen::Vector4d& mean,
  const Eigen::MatrixXd& covariance,
  const std::vector<Euler> &axes) :
    fuse_core::Constraint{orientation.uuid()},
    mean_(mean),
    axes_(axes)
{
  assert(axes_.size() > 0);
  axes_.erase(std::unique(axes_.begin(), axes_.end()), axes_.end());
  assert(axes_.size() <= 3);

  if (axes_.size() == 3)
  {
    sqrt_information_ = covariance.inverse().llt().matrixU();
  }
  else
  {
    assert(covariance.rows() == static_cast<int>(axes.size()));
    assert(covariance.cols() == static_cast<int>(axes.size()));

    // Compute the sqrt information of the provided cov matrix
    Eigen::MatrixXd partial_sqrt_information = covariance.inverse().llt().matrixU();

    // Assemble a mean vector and sqrt information matrix from the provided values, but in proper Variable order
    // What are we doing here?
    // The constraint equation is defined as: cost(x) = ||A * (x - b)||^2
    // If we are measuring a subset of dimensions, we only want to produce costs for the measured dimensions.
    // But the variable vectors will be full sized. We can make this all work out by creating a non-square A
    // matrix, where each row computes a cost for one measured dimensions, and the columns are in the order
    // defined by the variable.
    sqrt_information_ = Eigen::MatrixXd::Zero(axes.size(), 3);
    for (size_t r = 0; r < size_t(partial_sqrt_information.rows()); ++r)
    {
      for (size_t c = 0; c < size_t(partial_sqrt_information.cols()); ++c)
      {
        const size_t row_ind = size_t(axes[r]) - size_t(Euler::ROLL);
        const size_t col_ind = size_t(axes[c]) - size_t(Euler::ROLL);

        sqrt_information_(row_ind, col_ind) = partial_sqrt_information(r, c);
      }
    }
  }
}

AbsoluteOrientation3DStampedConstraint::AbsoluteOrientation3DStampedConstraint(
  const fuse_variables::Orientation3DStamped& orientation,
  const Eigen::Quaterniond& mean,
  const Eigen::MatrixXd& covariance,
  const std::vector<Euler> &axes) :
    AbsoluteOrientation3DStampedConstraint(orientation, toEigen(mean), covariance, axes)
{
}

AbsoluteOrientation3DStampedConstraint::AbsoluteOrientation3DStampedConstraint(
  const fuse_variables::Orientation3DStamped& orientation,
  const geometry_msgs::Quaternion& mean,
  const std::array<double, 9>& covariance,
  const std::vector<Euler> &axes) :
    AbsoluteOrientation3DStampedConstraint(orientation, toEigen(mean), toEigen(covariance), axes)
{
}

Eigen::MatrixXd AbsoluteOrientation3DStampedConstraint::covariance() const
{
  if (axes_.size() == 3)
  {
    return (sqrt_information_.transpose() * sqrt_information_).inverse();
  }
  else
  {
    // We want to compute:
    // cov = (sqrt_info' * sqrt_info)^-1
    // With some linear algebra, we can swap the transpose and the inverse.
    // cov = (sqrt_info^-1) * (sqrt_info^-1)'
    // But sqrt_info _may_ not be square. So we need to compute the pseudoinverse instead.
    // Eigen doesn't have a pseudoinverse function (for probably very legitimate reasons).
    // So we set the right hand side to identity, then solve using one of Eigen's many decompositions.
    auto I = Eigen::MatrixXd::Identity(sqrt_information_.rows(), sqrt_information_.cols());
    Eigen::MatrixXd pinv = sqrt_information_.colPivHouseholderQr().solve(I);
    return pinv * pinv.transpose();
  }
}

void AbsoluteOrientation3DStampedConstraint::print(std::ostream& stream) const
{
  stream << type() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  orientation variable: " << variables_.at(0) << "\n"
         << "  mean: " << mean_.transpose() << "\n"
         << "  sqrt_info: " << sqrtInformation() << "\n";
}

fuse_core::Constraint::UniquePtr AbsoluteOrientation3DStampedConstraint::clone() const
{
  return AbsoluteOrientation3DStampedConstraint::make_unique(*this);
}

ceres::CostFunction* AbsoluteOrientation3DStampedConstraint::costFunction() const
{
  //return new ceres::AutoDiffCostFunction<NormalPriorOrientation3DCostFunctor, 3, 4>(
  //   new NormalPriorOrientation3DCostFunctor(sqrt_information_, mean_, axes_));

  if (axes_.size() == 3)
  {
    return new ceres::AutoDiffCostFunction<NormalPriorOrientation3DCostFunctor, 3, 4>(
      new NormalPriorOrientation3DCostFunctor(sqrt_information_, mean_, axes_));
  }
  else if(axes_.size() == 2)
  {
    return new ceres::AutoDiffCostFunction<NormalPriorOrientation3DCostFunctor, 2, 4>(
      new NormalPriorOrientation3DCostFunctor(sqrt_information_, mean_, axes_));
  }
  else // if(axes_.size() == 1)
  {
    return new ceres::AutoDiffCostFunction<NormalPriorOrientation3DCostFunctor, 1, 4>(
      new NormalPriorOrientation3DCostFunctor(sqrt_information_, mean_, axes_));
  }
}

Eigen::Vector4d AbsoluteOrientation3DStampedConstraint::toEigen(const Eigen::Quaterniond& quaternion)
{
  Eigen::Vector4d eigen_quaternion_vector;
  eigen_quaternion_vector << quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z();
  return eigen_quaternion_vector;
}

Eigen::Vector4d AbsoluteOrientation3DStampedConstraint::toEigen(const geometry_msgs::Quaternion& quaternion)
{
  Eigen::Vector4d eigen_quaternion_vector;
  eigen_quaternion_vector << quaternion.w, quaternion.x, quaternion.y, quaternion.z;
  return eigen_quaternion_vector;
}

Eigen::Matrix3d AbsoluteOrientation3DStampedConstraint::toEigen(const std::array<double, 9> covariance)
{
  return Eigen::Matrix3d(covariance.data());
}

}  // namespace fuse_constraints
