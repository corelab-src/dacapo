
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_COVERAGERECORDER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

#define DEBUG_TYPE "dacapo"

namespace {
/// Pass to bufferize Arith ops.
struct CoverageRecorderPass
    : public hecate::earth::impl::CoverageRecorderBase<CoverageRecorderPass> {
  CoverageRecorderPass() {}
  CoverageRecorderPass(hecate::earth::CoverageRecorderOptions ops) {}
  CoverageRecorderPass(std::pair<int64_t, int64_t> ops) {
    this->waterline = ops.first;
    this->threshold = ops.second;
  }

  void runOnOperation() override {

    auto func = getOperation();
    auto dup = func.clone();
    auto &&from = dup->getAttrOfType<mlir::DenseI64ArrayAttr>("cutted_edge")
                      .asArrayRef()
                      .front();

    auto &&block = dup.getBody().front();
    auto &&operations = block.getOperations();
    mlir::OpBuilder builder(dup);

    // Set function argument types
    auto &&inputType_attrs = dup->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    for (size_t i = 0; i < dup.getNumArguments(); i++) {
      auto argval = dup.getArgument(i);
      auto input_type = inputType_attrs[i]
                            .dyn_cast<mlir::TypeAttr>()
                            .getValue()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>();
      argval.setType(input_type);
    }

    // Calculate the coverages
    int64_t coverage = -1, bootCoverage = -1;
    for (auto &&op : operations) {
      if (auto sop = dyn_cast<hecate::earth::ForwardMgmtInterface>(op)) {
        auto opid = hecate::getIntegerAttr("opid", sop->getResult(0));
        if (!isa<hecate::earth::BootstrapOp>(sop) && opid < from)
          continue;

        // PARS Scale Management
        builder.setInsertionPointAfter(sop.getOperation());
        sop.processOperandsPARS(waterline);
        /* sop.processOperandsEVA(waterline); */
        inferTypeForward(sop);
        sop.processResultsPARS(waterline);
        /* sop.processResultsEVA(waterline); */
        /////////////////////////////////////////

        // Find Bootstrapping Coverage
        if (bootCoverage < 0 && !sop.isBootstrappable()) {
          bootCoverage = opid;
          continue;
        }

        // Find Coverage
        if (!sop.isValidated()) {
          coverage = opid;
          break;
        }
      }
    }
    /* llvm::errs() << "from End: " << from << " cv : " << coverage */
    /*              << " bc : " << bootCoverage << '\n'; */
    dup.erase();
    func->setAttr("coverages",
                  builder.getDenseI64ArrayAttr({coverage, bootCoverage}));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
