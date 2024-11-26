
#ifndef HECATE_ANALYSIS_CANDIDATEANALYSIS
#define HECATE_ANALYSIS_CANDIDATEANALYSIS

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallSet.h"
#include <set>

namespace hecate {

struct ValueInfo {

public:
  ValueInfo(int64_t opid, mlir::Value v);
  mlir::Value getValue() const;

  mlir::SmallVector<int64_t, 4> getLiveIns() const;
  mlir::SmallVector<int64_t, 4> getLiveOuts() const;
  bool isBypassEdge(int64_t to) const;
  int64_t getCoverage() const;
  int64_t getBootCoverage() const;
  int64_t getDeadOpid() const;
  int64_t getThresholdOpid() const;
  mlir::SmallVector<int64_t, 4> getValidLiveOuts() const;

  void setLiveOuts(mlir::SmallVector<int64_t, 4> outs);
  void setLiveIns(mlir::SmallVector<int64_t, 4> ins);
  void setThresholdOpid(int64_t OverThr);
  void setBypassEdge();
  void setCoverage(int64_t);
  void setBootCoverage(int64_t);
  void setDeadOpid(int64_t);
  void setValidLiveOuts(mlir::SmallVector<int64_t, 4> outs);

private:
  int64_t opid;
  mlir::Operation *_op;
  mlir::Value v;

  // Liveness Information
  mlir::SmallVector<int64_t, 4> liveOuts;
  mlir::SmallVector<int64_t, 4> liveIns;
  int64_t deadOpid;

  // Coverage Information
  bool isBypass = false;
  int64_t bootCoverage = -1;
  int64_t coverage = -1;
  int64_t thresholdOpid = std::numeric_limits<int64_t>::max();
  // except bypassed edges
  mlir::SmallVector<int64_t, 4> validLiveOuts;
};

struct CandidateAnalysis {
public:
  CandidateAnalysis(mlir::Operation *op);

  // Default Implementation
  int64_t getOpid(mlir::Value v) const;
  int64_t getOpid(mlir::Operation *op) const;
  ValueInfo *getValueInfo(int64_t opid);

  mlir::SmallVector<int64_t, 4> getTargets(int64_t opid) const;
  mlir::SmallVector<int64_t, 4> getTargets(int64_t opid, int64_t setNum) const;
  mlir::SmallVector<int64_t, 4> getTargets(int64_t from, int64_t to,
                                           int64_t setNum) const;
  mlir::SmallVector<int64_t, 4> getCandidateSet(size_t size) const;
  mlir::SmallVector<int64_t, 4> getCandidates() const;
  mlir::SmallVector<int64_t, 4> getEdges() const;
  mlir::SmallVector<bool, 4> getBypassTypeOfLiveOuts(int64_t opid);

  size_t getMaxNumOuts() const;
  size_t getNumValues() const;
  size_t getNumCandidates() const;
  size_t getNumEdges() const;
  int64_t getRetOpid() const;
  ScaleManagementUnit getSMU() const;

  void setBypassEdges(int64_t from, int64_t OverThr);
  void sortValidCandidates(int64_t opid);
  void finalizeCandidates(int64_t setNum);
  void pushFromCoverage(int64_t from, mlir::SmallVector<int64_t, 2> coverages);
  mlir::SmallVector<int64_t, 4> sortTargets(int64_t setNum);
  mlir::SmallVector<int64_t, 4>
  sortTargets(int64_t setNum, mlir::SmallVector<int64_t, 4> opids);
  std::map<int64_t, mlir::SmallVector<int64_t, 4>> toFromMap;

private:
  mlir::SmallVector<int64_t, 4> candidates;
  mlir::SmallVector<int64_t, 4> edges;
  llvm::DenseMap<size_t, mlir::SmallVector<int64_t, 4>> candidateSet;
  llvm::DenseMap<std::pair<int64_t, int64_t>, mlir::SmallVector<int64_t, 4>>
      segToBypasses;

  llvm::SmallVector<ValueInfo, 4> values;
  int64_t retOpid;
  mlir::Liveness _l;
  mlir::Operation *_op;
  ScaleManagementUnit smu;
};
} // namespace hecate

#endif
