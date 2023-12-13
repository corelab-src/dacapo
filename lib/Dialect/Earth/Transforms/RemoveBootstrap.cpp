
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_REMOVEBOOTSTRAP
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct RemoveBootstrapPass
    : public hecate::earth::impl::RemoveBootstrapBase<RemoveBootstrapPass> {
  RemoveBootstrapPass() {}

  void runOnOperation() override {

    auto func = getOperation();
    func.walk([&](hecate::earth::BootstrapOp bop) {
      bop.replaceAllUsesWith(bop.getOperand());
      bop.erase();
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
