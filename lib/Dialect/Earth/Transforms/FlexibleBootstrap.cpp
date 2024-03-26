
//===- Bufferize.cpp - Bufferization for Arith ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_FLEXIBLEBOOTSTRAP
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "hecate_em"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct FlexibleBootstrapPass
    : public hecate::earth::impl::FlexibleBootstrapBase<FlexibleBootstrapPass> {
  FlexibleBootstrapPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    auto &&block = func.getBody().front();
    auto &&operations = block.getOperations();
    // TODO: walk function has error for bootstrap operation
    /* func.walk([&](hecate::earth::BootstrapOp bop) { */
    uint64_t minModFactor = -1;
    for (auto &&op : operations) {
      if (auto bop = dyn_cast<hecate::earth::BootstrapOp>(op)) {

        for (auto &&oper : bop.getResult().getUsers()) {
          if (auto oop = dyn_cast<hecate::earth::ModswitchOp>(oper)) {
            minModFactor = std::min(minModFactor, oop.getDownFactor());
          } else
            minModFactor = 0;
        }
        // Check that every user needs the "downFactor"ed level
        if (!minModFactor) {
          return; // Go to next operation
        }
        bop.setTargetLevel(bop.getTargetLevel() + minModFactor);
        bop.getResult()
            .getType()
            .dyn_cast<hecate::earth::HEScaleTypeInterface>()
            .switchLevel(bop.getTargetLevel());

        // Change the user modswitch downFactors
        for (auto &&oper : bop.getResult().getUsers()) {
          if (auto oop = dyn_cast<hecate::earth::ModswitchOp>(oper)) {
            auto newDownFactor = oop.getDownFactor() - minModFactor;
            if (!newDownFactor) {
              oop.replaceAllUsesWith(oop.getOperand());
              oop.erase();
            } else {
              oop.setDownFactor(oop.getDownFactor() - minModFactor);
            }
          }
        }
        /* }); */
        /* LLVM_DEBUG(llvm::dbgs() << __FILE__ << ":" << __LINE__ << "\n"); */
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
