
#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
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
#include "mlir/IR/BuiltinAttributes.h"
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
    /* auto &ca = getAnalysis<hecate::CandidateAnalysis>(); */
    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    auto values = hecate::earth::attachOpid(func);
    auto &&cutted_edges =
        func->getAttrOfType<mlir::DenseI64ArrayAttr>("cutted_edge")
            .asArrayRef();
    /* auto &&from = cutted_edges.front(); */
    auto &&to = cutted_edges.back();

    // set liveout operations of end-edge as return operands
    auto &&ret = func->getAttrOfType<mlir::DenseI64ArrayAttr>("segment_return")
                     .asArrayRef();
    mlir::SmallVector<mlir::Value, 4> rets;
    if (ret.empty()) {
      func->setAttr("is_mid_segment", builder.getBoolAttr(false));
    } else {
      for (uint64_t i = 0; i < ret.size(); i++) {
        func->walk([&](hecate::earth::BootstrapOp bop) {
          auto btp = hecate::getIntegerAttr("opid", bop);
          /* values.push_back(val); */
          if (ret[i] == btp) {
            rets.push_back(bop);
            return;
          }
        });
        if (rets.size() == i)
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
    /* llvm::errs() << "bypassTyope.size" << bypassTypes.size() << '\n'; */
    for (uint64_t i = 0; i < inputs.size(); i++) {
      auto &&target = values[inputs[i]];
      /* target.dump(); */
      /* llvm::errs() << "check bypass : " << bypassTypes.size() << '\n'; */
      /* auto &&isBypass = bypassTypes[i].dyn_cast<mlir::BoolAttr>().getValue();
       */
      auto arg = func.front().addArgument(target.getType(), func.getLoc());
      /* hecate::setIntegerAttr("is_bypassed", arg, isBypass); */
      rewriter.replaceAllUsesWith(target, arg);
      inputTypes.push_back(arg.getType());
    }

    // set Function Type
    for (auto ret : func.getRegion().front().getTerminator()->getOperands()) {
      /* auto opid = hecate::getIntegerAttr("opid", ret); */
      retTypes.push_back(ret.getType());
      /* hecate::setIntegerAttr("is_bypassed", ret, */
      /*                        ca.getValueInfo(opid)->isBypassEdge(to)); */
    }
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
