#include <stdexcept>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "world.hpp"

namespace {

bool jcurSttConfigured(const Ferromagnet* magnet) {
  return !magnet->jcur_stt.assuredZero();
}

bool jcurZlConfigured(const Ferromagnet* magnet) {
  return !magnet->jcur_zl.assuredZero();
}

bool jcurLegacyConfigured(const Ferromagnet* magnet) {
  return !magnet->jcur.assuredZero();
}

const Parameter& zhangLiPolParam(const Ferromagnet* magnet) {
  return magnet->pol_zl.assuredZero() ? magnet->pol : magnet->pol_zl;
}

const Parameter& zhangLiXiParam(const Ferromagnet* magnet) {
  return magnet->xi_zl.assuredZero() ? magnet->xi : magnet->xi_zl;
}

void validateCombinedSttConfig(const Ferromagnet* magnet) {
  if (!magnet->enableCombinedSpinTransferTorque) {
    return;
  }

  const bool splitStt = jcurSttConfigured(magnet);
  const bool splitZl = jcurZlConfigured(magnet);
  const bool legacy = jcurLegacyConfigured(magnet);

  if (legacy && (splitStt || splitZl)) {
    throw std::invalid_argument(
        "Combined spin-transfer torque: cannot use jcur together with "
        "jcur_stt and/or jcur_zl.");
  }

  if (splitStt != splitZl) {
    throw std::invalid_argument(
        "Combined spin-transfer torque: jcur_stt and jcur_zl must both be "
        "set when using split current densities.");
  }
}

struct ResolvedSttCurrents {
  const VectorParameter* zhangLi;
  const VectorParameter* slonczewski;
};

ResolvedSttCurrents resolveSttCurrents(const Ferromagnet* magnet) {
  validateCombinedSttConfig(magnet);

  ResolvedSttCurrents resolved{&magnet->jcur, &magnet->jcur};

  if (magnet->enableCombinedSpinTransferTorque) {
    if (jcurSttConfigured(magnet) && jcurZlConfigured(magnet)) {
      resolved.slonczewski = &magnet->jcur_stt;
      resolved.zhangLi = &magnet->jcur_zl;
    }
  }

  return resolved;
}

bool zhangLiSTTAssuredZero(const Ferromagnet* magnet,
                           const VectorParameter& jcur,
                           const Parameter& pol) {
  return !magnet->enableZhangLiTorque || magnet->msat.assuredZero() ||
         jcur.assuredZero() || pol.assuredZero();
}

bool slonczewskiSTTAssuredZero(const Ferromagnet* magnet,
                               const VectorParameter& jcur) {
  return !magnet->enableSlonczewskiTorque ||
         magnet->msat.assuredZero() || jcur.assuredZero() ||
         magnet->freeLayerThickness.assuredZero() ||
         magnet->fixedLayer.assuredZero() ||
         (magnet->epsilonPrime.assuredZero() &&
          (magnet->Lambda.assuredZero() || magnet->pol.assuredZero()));
}

// Zhang-Li and Slonczewski contributions are accumulated in torque/gamma
// (effective-field units, matching mumax3 STTorque). They are multiplied by
// gamma once in k_SpinTransferTorque to obtain dm/dt in rad/s for evalTorque.

__device__ real3 zhangLiTorqueAtCell(int idx, const CuField mField,
                                     const CuParameter msatParam,
                                     const CuParameter polParam,
                                     const CuParameter xiParam,
                                     const CuParameter alphaParam,
                                     const CuParameter gammaParam,
                                     const CuVectorParameter jcurParam,
                                     const Grid grid,
                                     const Grid mastergrid,
                                     const real3 cellsize) {
  real3 m = mField.vectorAt(idx);

  const real3 j = jcurParam.vectorAt(idx);
  const real msat = msatParam.valueAt(idx);
  const real pol = polParam.valueAt(idx);
  const real xi = xiParam.valueAt(idx);
  const real alpha = alphaParam.valueAt(idx);
  const real gamma = gammaParam.valueAt(idx);

  if (msat == 0 || pol == 0 || j == real3{0, 0, 0} || gamma == 0) {
    return real3{0, 0, 0};
  }

  const int3 coo = grid.index2coord(idx);

  // mumax3: b = (1/msat) * MUB/(2*QE*gamma) / (1+xi^2), hspin = (b/cellsize)*J*dm
  const real3 u = MUB * pol / (QE * msat * (1 + xi * xi) * 2 * gamma) * j;

  real3 hspin{0, 0, 0};

  for (int sign : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{sign, 0, 0});
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.x * m_ / (2 * cellsize.x);
    }
  }
  for (int sign : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{0, sign, 0});
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.y * m_ / (2 * cellsize.y);
    }
  }
  for (int sign : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{0, 0, sign});
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.z * m_ / (2 * cellsize.z);
    }
  }

  const real3 mxh = cross(m, hspin);
  const real3 mxmxh = cross(m, mxh);
  return (-1 / (1 + alpha * alpha)) *
         ((1 + xi * alpha) * mxmxh + (xi - alpha) * mxh);
}

__device__ real3 slonczewskiTorqueAtCell(int idx, const CuField mField,
                                          const CuParameter msatParam,
                                          const CuParameter polParam,
                                          const CuParameter lambdaParam,
                                          const CuParameter alphaParam,
                                          const CuVectorParameter jcurParam,
                                          const CuParameter epsilonPrime,
                                          const CuVectorParameter fixedLayer,
                                          const CuParameter freeLayerThickness,
                                          const bool fixedLayerOnTop) {
  real3 m = mField.vectorAt(idx);

  const real3 jj = jcurParam.vectorAt(idx);
  const real jz = jj.z;

  const real msat = msatParam.valueAt(idx);
  const real pol = polParam.valueAt(idx);
  const real alpha = alphaParam.valueAt(idx);

  const real3 p = fixedLayer.vectorAt(idx);
  const real lambda = lambdaParam.valueAt(idx);
  const real eps_p = epsilonPrime.valueAt(idx);
  real d = freeLayerThickness.valueAt(idx);
  if (!fixedLayerOnTop) {
    d *= -1;
  }

  if (msat == 0 || jz == 0 || d == 0 || p == real3{0, 0, 0} ||
      (eps_p == 0 && (lambda == 0 || pol == 0))) {
    return real3{0, 0, 0};
  }

  const real beta = (HBAR / QE) * jz / (msat * d);
  const real lambda2 = lambda * lambda;
  const real eps = pol * lambda2 / ((lambda2 + 1) + (lambda2 - 1) * dot(m, p));
  const real gilb = 1 / (1 + alpha * alpha);

  const real3 pxm = cross(p, m);
  const real3 mxpxm = cross(m, pxm);
  const real mxpxmFac = gilb * (beta * eps + alpha * beta * eps_p);
  const real pxmFac = gilb * (beta * eps_p - alpha * beta * eps);
  return mxpxmFac * mxpxm + pxmFac * pxm;
}

__global__ void k_SpinTransferTorque(
    CuField torque, const CuField mField, const CuParameter msatParam,
    const CuParameter polZlParam, const CuParameter polSttParam,
    const CuParameter lambdaParam, const CuParameter alphaParam,
    const CuParameter gammaParam, const CuParameter xiParam,
    const CuParameter epsilonPrime, const CuVectorParameter jcurZlParam,
    const CuVectorParameter jcurSttParam, const CuVectorParameter fixedLayer,
    const CuParameter freeLayerThickness, const CuParameter frozenSpins,
    const bool fixedLayerOnTop, const bool enableZhangLi,
    const bool enableSlonczewski, const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!torque.cellInGrid(idx)) {
    return;
  }

  if (!torque.cellInGeometry(idx) || (frozenSpins.valueAt(idx) != 0)) {
    torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const Grid grid = torque.system.grid;
  const real3 cellsize = torque.system.cellsize;

  real3 t{0, 0, 0};

  if (enableZhangLi) {
    t += zhangLiTorqueAtCell(idx, mField, msatParam, polZlParam, xiParam,
                             alphaParam, gammaParam, jcurZlParam, grid,
                             mastergrid, cellsize);
  }

  if (enableSlonczewski) {
    t += slonczewskiTorqueAtCell(idx, mField, msatParam, polSttParam, lambdaParam,
                                 alphaParam, jcurSttParam, epsilonPrime, fixedLayer,
                                 freeLayerThickness, fixedLayerOnTop);
  }

  torque.setVectorInCell(idx, gammaParam.valueAt(idx) * t);
}

}  // namespace

bool ZhangLiSTTAssuredZero(const Ferromagnet* magnet) {
  const auto currents = resolveSttCurrents(magnet);
  const Parameter& pol =
      magnet->enableCombinedSpinTransferTorque ? zhangLiPolParam(magnet)
                                               : magnet->pol;
  return zhangLiSTTAssuredZero(magnet, *currents.zhangLi, pol);
}

bool SlonczewskiSTTAssuredZero(const Ferromagnet* magnet) {
  const auto currents = resolveSttCurrents(magnet);
  return slonczewskiSTTAssuredZero(magnet, *currents.slonczewski);
}

bool spinTransferTorqueAssuredZero(const Ferromagnet* magnet) {
  if (magnet->enableCombinedSpinTransferTorque) {
    return ZhangLiSTTAssuredZero(magnet) && SlonczewskiSTTAssuredZero(magnet);
  }
  return ZhangLiSTTAssuredZero(magnet) && SlonczewskiSTTAssuredZero(magnet);
}

Field evalSpinTransferTorque(const Ferromagnet* magnet) {
  Field torque(magnet->system(), 3);

  if (spinTransferTorqueAssuredZero(magnet)) {
    torque.makeZero();
    return torque;
  }

  const auto currents = resolveSttCurrents(magnet);
  const Parameter& polZl =
      magnet->enableCombinedSpinTransferTorque ? zhangLiPolParam(magnet)
                                               : magnet->pol;
  const Parameter& xiZl =
      magnet->enableCombinedSpinTransferTorque ? zhangLiXiParam(magnet)
                                               : magnet->xi;

  const bool zhangLiActive =
      !zhangLiSTTAssuredZero(magnet, *currents.zhangLi, polZl);
  const bool slonczewskiActive =
      !slonczewskiSTTAssuredZero(magnet, *currents.slonczewski);

  if (!zhangLiActive && !slonczewskiActive) {
    torque.makeZero();
    return torque;
  }

  const int ncells = magnet->grid().ncells();
  auto m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto pol = magnet->pol.cu();
  auto xi = xiZl.cu();
  auto alpha = magnet->alpha.cu();
  auto gamma = magnet->gamma.cu();
  auto jcurZl = currents.zhangLi->cu();
  auto jcurStt = currents.slonczewski->cu();
  auto lambda = magnet->Lambda.cu();
  auto epsilonPrime = magnet->epsilonPrime.cu();
  auto fixedLayer = magnet->fixedLayer.cu();
  auto freeLayerThickness = magnet->freeLayerThickness.cu();
  const bool fixedLayerOnTop = magnet->fixedLayerOnTop;
  auto frozenSpins = magnet->frozenSpins.cu();
  auto polZlCu = polZl.cu();
  auto polSttCu = magnet->pol.cu();

  bool enableZhangLi = zhangLiActive;
  bool enableSlonczewski = slonczewskiActive;

  if (!magnet->enableCombinedSpinTransferTorque) {
    if (slonczewskiActive) {
      enableZhangLi = false;
    } else {
      enableSlonczewski = false;
    }
  }

  cudaLaunch(ncells, k_SpinTransferTorque, torque.cu(), m, msat, polZlCu, polSttCu,
             lambda, alpha, gamma, xi, epsilonPrime, jcurZl, jcurStt,
             fixedLayer, freeLayerThickness, frozenSpins, fixedLayerOnTop,
             enableZhangLi, enableSlonczewski,
             magnet->world()->mastergrid());

  return torque;
}

FM_FieldQuantity spinTransferTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalSpinTransferTorque, 3, "spintransfer_torque",
                          "rad/s");
}
