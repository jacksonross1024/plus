#pragma once

#include <cmath>

#ifdef __CUDACC__
#define POISSON_HD __host__ __device__ inline
#else
#define POISSON_HD inline
#endif

// SPD conductivity (upper triangle) and Hall skew used by the FV assembly.
struct PoissonSym6 {
  float xx = 0.0f;
  float yy = 0.0f;
  float zz = 0.0f;
  float xy = 0.0f;
  float xz = 0.0f;
  float yz = 0.0f;
};

struct PoissonSkew3 {
  // [[ 0, xy, xz],
  //  [-xy, 0, yz],
  //  [-xz,-yz, 0]]
  float xy = 0.0f;
  float xz = 0.0f;
  float yz = 0.0f;
};

struct PoissonConductivitySplit {
  PoissonSym6 S;
  PoissonSkew3 K;
};

struct PoissonConductivityInputs {
  float sigma0 = 0.0f;
  bool is_fm = false;
  bool is_pt = false;
  bool amr_enabled = false;
  bool ahe_enabled = false;
  bool the_enabled = false;
  bool ohe_enabled = false;
  bool resistivity_invert = false;
  double amr_ratio = 0.0;
  double ahe_ratio = 0.0;
  double the_ratio = 0.0;
  double hall_coefficient = 0.0;
  float mx = 0.0f;
  float my = 0.0f;
  float mz = 0.0f;
  float hx = 0.0f;
  float hy = 0.0f;
  float hz = 0.0f;
  float bx = 0.0f;
  float by = 0.0f;
  float bz = 0.0f;
};

POISSON_HD PoissonSym6 poisson_amr_conductivity(float sigma0,
                                                double amr_ratio,
                                                float mx,
                                                float my,
                                                float mz) {
  const double q = 6.0 * amr_ratio / (6.0 + amr_ratio);
  const double s = static_cast<double>(sigma0);
  return {static_cast<float>(s * (1.0 - q * (static_cast<double>(mx) * mx - 1.0 / 3.0))),
          static_cast<float>(s * (1.0 - q * (static_cast<double>(my) * my - 1.0 / 3.0))),
          static_cast<float>(s * (1.0 - q * (static_cast<double>(mz) * mz - 1.0 / 3.0))),
          static_cast<float>(-s * q * static_cast<double>(mx) * my),
          static_cast<float>(-s * q * static_cast<double>(mx) * mz),
          static_cast<float>(-s * q * static_cast<double>(my) * mz)};
}

POISSON_HD bool poisson_invert_3x3(const double m[9], double inv[9]) {
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (!(det > 1e-30) && !(det < -1e-30)) {
    return false;
  }
  const double idet = 1.0 / det;
  inv[0] = (m[4] * m[8] - m[5] * m[7]) * idet;
  inv[1] = (m[2] * m[7] - m[1] * m[8]) * idet;
  inv[2] = (m[1] * m[5] - m[2] * m[4]) * idet;
  inv[3] = (m[5] * m[6] - m[3] * m[8]) * idet;
  inv[4] = (m[0] * m[8] - m[2] * m[6]) * idet;
  inv[5] = (m[2] * m[3] - m[0] * m[5]) * idet;
  inv[6] = (m[3] * m[7] - m[4] * m[6]) * idet;
  inv[7] = (m[1] * m[6] - m[0] * m[7]) * idet;
  inv[8] = (m[0] * m[4] - m[1] * m[3]) * idet;
  return true;
}

POISSON_HD PoissonConductivitySplit poisson_split_from_matrix(const double sig[9]) {
  PoissonConductivitySplit out;
  out.S.xx = static_cast<float>(sig[0]);
  out.S.yy = static_cast<float>(sig[4]);
  out.S.zz = static_cast<float>(sig[8]);
  out.S.xy = static_cast<float>(0.5 * (sig[1] + sig[3]));
  out.S.xz = static_cast<float>(0.5 * (sig[2] + sig[6]));
  out.S.yz = static_cast<float>(0.5 * (sig[5] + sig[7]));
  out.K.xy = static_cast<float>(0.5 * (sig[1] - sig[3]));
  out.K.xz = static_cast<float>(0.5 * (sig[2] - sig[6]));
  out.K.yz = static_cast<float>(0.5 * (sig[5] - sig[7]));
  return out;
}

POISSON_HD PoissonConductivitySplit poisson_additive_conductivity(
    const PoissonConductivityInputs& in) {
  PoissonConductivitySplit out;
  const float s = in.sigma0;
  if (!(s > 1e-20f)) {
    return out;
  }
  if (in.amr_enabled && in.is_fm) {
    out.S = poisson_amr_conductivity(s, in.amr_ratio, in.mx, in.my, in.mz);
  } else {
    out.S = {s, s, s, 0.0f, 0.0f, 0.0f};
  }
  if (in.ohe_enabled && (in.is_pt || in.is_fm)) {
    const double sigma_oh = -static_cast<double>(s) * static_cast<double>(s) * in.hall_coefficient;
    out.K.xy += static_cast<float>(sigma_oh * static_cast<double>(-in.bz));
    out.K.xz += static_cast<float>(sigma_oh * static_cast<double>(in.by));
    out.K.yz += static_cast<float>(sigma_oh * static_cast<double>(-in.bx));
  }
  if (in.is_fm) {
    if (in.ahe_enabled && !(in.mx == 0.0f && in.my == 0.0f && in.mz == 0.0f)) {
      const float sigma_ahe = static_cast<float>(in.ahe_ratio * static_cast<double>(s));
      out.K.xy += -sigma_ahe * in.mz;
      out.K.xz += sigma_ahe * in.my;
      out.K.yz += -sigma_ahe * in.mx;
    }
    if (in.the_enabled) {
      const float sigma_the = static_cast<float>(in.the_ratio * static_cast<double>(s));
      out.K.xy += -sigma_the * in.hz;
      out.K.xz += sigma_the * in.hy;
      out.K.yz += -sigma_the * in.hx;
    }
  }
  return out;
}

POISSON_HD PoissonConductivitySplit poisson_resistivity_inverse(
    const PoissonConductivityInputs& in) {
  const float s = in.sigma0;
  if (!(s > 1e-20f)) {
    return {};
  }
  const double sigma0 = static_cast<double>(s);
  const double rho0 = 1.0 / sigma0;

  double rho_sym[9] = {rho0, 0.0, 0.0, 0.0, rho0, 0.0, 0.0, 0.0, rho0};
  if (in.amr_enabled && in.is_fm && !(in.mx == 0.0f && in.my == 0.0f && in.mz == 0.0f)) {
    const PoissonSym6 samr = poisson_amr_conductivity(s, in.amr_ratio, in.mx, in.my, in.mz);
    const double sm[9] = {static_cast<double>(samr.xx), static_cast<double>(samr.xy),
                          static_cast<double>(samr.xz), static_cast<double>(samr.xy),
                          static_cast<double>(samr.yy), static_cast<double>(samr.yz),
                          static_cast<double>(samr.xz), static_cast<double>(samr.yz),
                          static_cast<double>(samr.zz)};
    double rho_amr[9];
    if (poisson_invert_3x3(sm, rho_amr)) {
      for (int i = 0; i < 9; ++i) {
        rho_sym[i] = rho_amr[i];
      }
    }
  }

  double wx = 0.0;
  double wy = 0.0;
  double wz = 0.0;
  if (in.ohe_enabled && (in.is_pt || in.is_fm)) {
    wx += in.hall_coefficient * static_cast<double>(in.bx);
    wy += in.hall_coefficient * static_cast<double>(in.by);
    wz += in.hall_coefficient * static_cast<double>(in.bz);
  }
  if (in.is_fm) {
    if (in.ahe_enabled) {
      // Minus: additive CUDA AHE is J = σ E + σ_AH m × E, which is the inverse of
      // E = ρ J - R_s m × J. Using +R_s m here would flip the Hall voltage.
      const double r_s = in.ahe_ratio / sigma0;
      wx -= r_s * static_cast<double>(in.mx);
      wy -= r_s * static_cast<double>(in.my);
      wz -= r_s * static_cast<double>(in.mz);
    }
    if (in.the_enabled) {
      const double r_th = in.the_ratio / sigma0;
      wx -= r_th * static_cast<double>(in.hx);
      wy -= r_th * static_cast<double>(in.hy);
      wz -= r_th * static_cast<double>(in.hz);
    }
  }

  // E = ρ_sym J + w × J, i.e. ρ = ρ_sym + [w]_× with
  // [w]_× = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]].
  const double rho[9] = {rho_sym[0],
                         rho_sym[1] - wz,
                         rho_sym[2] + wy,
                         rho_sym[3] + wz,
                         rho_sym[4],
                         rho_sym[5] - wx,
                         rho_sym[6] - wy,
                         rho_sym[7] + wx,
                         rho_sym[8]};
  double sig[9];
  if (!poisson_invert_3x3(rho, sig)) {
    return poisson_additive_conductivity(in);
  }
  return poisson_split_from_matrix(sig);
}

POISSON_HD PoissonConductivitySplit poisson_conductivity_from_inputs(
    const PoissonConductivityInputs& in) {
  if (in.resistivity_invert) {
    return poisson_resistivity_inverse(in);
  }
  return poisson_additive_conductivity(in);
}
