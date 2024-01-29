
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_CODESEGMENTATION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct CodeSegmentationPass
    : public hecate::earth::impl::CodeSegmentationBase<CodeSegmentationPass> {
  CodeSegmentationPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    auto values = hecate::earth::attachOpid(func);

    // set liveout operations of end-edge as return operands
    auto &&ret = func->getAttrOfType<mlir::DenseI64ArrayAttr>("segment_return")
                     .asArrayRef();
    mlir::SmallVector<mlir::Value, 4> rets;
    if (ret.empty()) {
      func->setAttr("is_mid_segment", builder.getBoolAttr(false));
    } else {
      for (uint64_t i = 0; i < ret.size(); i++) {
        rets.push_back(values[ret[i]]);
      }
      auto ter = func.front().getTerminator();
      ter->erase();
      builder.setInsertionPointToEnd(&func.front());
      builder.create<func::ReturnOp>(func.getLoc(), rets);
      func->setAttr("is_mid_segment", builder.getBoolAttr(true));
    }

    // set liveout operations of start-edge as function arguments
    auto &&inputs =
        func->getAttrOfType<mlir::DenseI64ArrayAttr>("segment_input")
            .asArrayRef();
    mlir::SmallVector<mlir::Type, 4> inputTypes, retTypes;
    for (auto arg : func.getArguments()) {
      inputTypes.push_back(arg.getType());
    }
    for (uint64_t i = 0; i < inputs.size(); i++) {
      auto &&target = values[inputs[i]];
      auto arg = func.front().addArgument(target.getType(), func.getLoc());
      rewriter.replaceAllUsesWith(target, arg);
      inputTypes.push_back(arg.getType());
    }

    // set Function Type
    for (auto ret : func.getRegion().front().getTerminator()->getOperands())
      retTypes.push_back(ret.getType());
    func.setFunctionType(builder.getFunctionType(inputTypes, retTypes));

    /* // DCE and Canonicalization */
    auto mod = func->getParentOfType<mlir::ModuleOp>();
    PassManager pm(mod.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    if (pm.run(mod).failed()) {
      assert(0 && "Pass Failed inside CodeSegmentation");
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
