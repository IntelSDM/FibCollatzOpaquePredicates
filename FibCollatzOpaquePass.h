#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct FibCollatzOpaquePass : llvm::PassInfoMixin<FibCollatzOpaquePass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunFibCollatzOpaque(llvm::Module& m);

extern llvm::cl::opt<bool> FibCollatzOpaqueEnable;