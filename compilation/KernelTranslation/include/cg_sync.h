#ifndef __CG_SYNC_H__
#define __CG_SYNC_H__

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <string>

bool isCGThreadBlockSync(const std::string &FuncNameStr);

bool isCGThreadGroupSync(const std::string &FuncNameStr);

bool isCGSync(const std::string &name);

bool isWarpSync(const std::string &FuncNameStr);

bool has_cg_group_sync(llvm::Module *M);

#endif // __CG_SYNC_H__
