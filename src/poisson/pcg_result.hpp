#pragma once

struct PcgResult {
  int iterations = 0;
  double initial_residual_max_norm = 0.0;
  double residual_max_norm = 0.0;
  double rhs_inf_norm = 0.0;
  double residual_relative = 0.0;
  bool converged = false;
  bool numerical_failure = false;
};
