
#ifndef HECATE_EARTH_TRANSFROMS_COMMON
#define HECATE_EARTH_TRANSFROMS_COMMON

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace hecate {
namespace earth {
void refineLevel(mlir::OpBuilder builder, mlir::Operation *op,
                 int64_t waterline, int64_t output_val, int64_t min_level);
void refineInputValues(mlir::func::FuncOp func, mlir::OpBuilder builder,
                       llvm::SmallVector<mlir::Type, 4> &inputTypes,
                       int64_t waterline, int64_t output_val);

void refineReturnValues(mlir::func::FuncOp func, mlir::OpBuilder builder,
                        llvm::SmallVector<mlir::Type, 4> inputTypes,
                        int64_t waterline, int64_t output_val);
void inferTypeForward(hecate::earth::ForwardMgmtInterface sop);

llvm::SmallVector<mlir::Value, 4> attachOpid(mlir::func::FuncOp func);

// To Calculate the waterline for SNR Rescaling
int64_t calcWaterline(hecate::ScaleManagementUnit &smu, mlir::Value v,
                      int64_t waterline);
int64_t calcWaterline(hecate::ScaleManagementUnit &smu, mlir::Operation *op,
                      int64_t waterline);

} // namespace earth
} // namespace hecate

#endif
