#include "FibCollatzOpaquePass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> FibCollatzOpaqueEnable("obffibcollpq", llvm::cl::init(true), llvm::cl::desc("Fibonacci + Collatz path-explosion opaque predicates."));

// cap per function emission at 8 because fib helper first call warm-up costs thousands of recursive invocations so stacking too many warm ups per function would inflate module load time; steady state after warm-up is O(1) per hit
static constexpr unsigned PerFunctionCap = 8;
static constexpr unsigned MinBBInsts = 4;

// fib's first-call warm up is thousands of recursive calls per invocation which inflates loop body cost and LLVM's loop analyses interact badly with mid-loop CFG splits
static constexpr bool SkipLoops = true;
static constexpr bool SkipEntry = false;

static constexpr unsigned BarrierPoolSize = 16;
static constexpr unsigned AuxMixCount = 3;
static constexpr unsigned FibWeightPct = 70;

static constexpr unsigned FibNMask = 0x7;
static constexpr unsigned FibNBase = 18;

// memo size must be a power of two so idx = n & (size-1) aliases cleanly without a modulo; 32 is the smallest power of two above the max n of 25
static constexpr unsigned FibMemoSize = 32;
static constexpr unsigned LucasMemoSize = 32;

struct PisanoEntry
{
    uint64_t M;
    uint64_t Pi;
    uint64_t MaxN;
};

// every pisano period in the pool stays at or under 16 so with n in [8, 15] the sum n+pi never exceeds 31, keeping fib-memo indices for n and n+pi distinct and avoiding slot-aliasing in the mod M predicate
static constexpr PisanoEntry PisanoPool[] = {
    { 2, 3, 28 },
    { 3, 8, 23 },
    { 4, 6, 25 },
    { 7, 16, 15 },
    { 8, 12, 19 },
    { 11, 10, 21 },
    { 21, 16, 15 },
    { 29, 14, 17 },
};
static constexpr unsigned PisanoPoolSize = sizeof(PisanoPool) / sizeof(PisanoPool[0]);

static constexpr unsigned PisanoActiveMin = 3;
static constexpr unsigned PisanoActiveMax = 4;

// collatz converges empirically within 500 steps across the entire n range so a cap of 2000 gives 4x headroom; cap exit is treated as the real path so a none converging input falls safe to real rather than crashing
static constexpr uint64_t CollatzInputMask = 0xFFFFFF;
static constexpr uint64_t CollatzSafetyCap = 2000;

static constexpr unsigned PolluteFloor = 8;
static constexpr unsigned PolluteCeiling = 48;
static constexpr unsigned DiamondCloneCap = 16;
static constexpr unsigned StealIntoHeadCeiling = 16;
static constexpr unsigned DiamondMinSuffix = 2;
static constexpr unsigned ConstPerturbationPct = 40;
static constexpr unsigned FibPFMin = 3;
static constexpr unsigned FibPFMax = 5;

static bool HasAnnotationTag(Function& f, const char* tag)
{
    Module* m = f.getParent();
    GlobalVariable* ga = m->getGlobalVariable("llvm.global.annotations");
    if (!ga || !ga->hasInitializer()) return false;
    auto* ca = dyn_cast<ConstantArray>(ga->getOperand(0));
    if (!ca) return false;
    for (unsigned i = 0; i < ca->getNumOperands(); ++i)
    {
        auto* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs || cs->getNumOperands() < 2) continue;
        if (cs->getOperand(0)->stripPointerCasts() != &f) continue;
        auto* annogv = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!annogv || !annogv->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataSequential>(annogv->getInitializer());
        if (!cds || !cds->isString()) continue;
        if (strcmp(cds->getAsString().str().c_str(), tag) == 0) return true;
    }
    return false;
}

static bool HasObfuscateAnnotation(Function& f)   { return HasAnnotationTag(f, "obfus"); }
static bool HasNoObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "noobfus"); }
static bool HasNoFibCollAnnotation(Function& f)   { return HasAnnotationTag(f, "nofibcollpq"); }

static std::mt19937_64 Rng;
static std::vector<GlobalVariable*> BarrierPool;
static std::vector<uint64_t> BarrierValues;
static std::string BuildSaltHex;
static Function* FibHelper = nullptr;
static GlobalVariable* FibMemo = nullptr;
static Function* LucasHelper = nullptr;
static GlobalVariable* LucasMemo = nullptr;
static GlobalVariable* SaltGV = nullptr;
static std::vector<unsigned> PisanoActive;

static bool IsPolluteBlock(BasicBlock* bb);
static bool IsFcpqInterior(BasicBlock* bb);
static void MarkFcpqInterior(BasicBlock* bb);

static void BuildBarrierPool(Module& m)
{
    BarrierPool.clear();
    BarrierValues.clear();
    BarrierPool.reserve(BarrierPoolSize);
    BarrierValues.reserve(BarrierPoolSize);

    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);

    for (unsigned i = 0; i < BarrierPoolSize; ++i)
    {
        uint64_t v = Rng();
        BarrierValues.push_back(v);

        std::string name = ".fcpq.barrier." + std::to_string(i);
        auto* gv = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, v), name);
        gv->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
        gv->setAlignment(Align(8));
        BarrierPool.push_back(gv);
    }
}

static Value* InsertBarrierLoad(IRBuilder<>& b, unsigned slotidx)
{
    if (BarrierPool.empty()) return nullptr;
    slotidx %= (unsigned)BarrierPool.size();
    GlobalVariable* gv = BarrierPool[slotidx];
    if (!gv) return nullptr;
    LoadInst* ld = b.CreateLoad(Type::getInt64Ty(b.getContext()), gv, true, "");
    ld->setAlignment(Align(8));
    return ld;
}

// the volatile i64 load blocks LLVM const prop from seeing through to the baked xor constant so the per site constant survives every downstream fold
static Value* InsertFoldBarrier(IRBuilder<>& b, uint64_t desiredconst)
{
    if (BarrierPool.empty()) return nullptr;
    const unsigned slotidx = (unsigned)(Rng() % BarrierPool.size());
    Value* ld = InsertBarrierLoad(b, slotidx);
    const uint64_t xork = BarrierValues[slotidx] ^ desiredconst;
    Value* r = b.CreateXor(ld, ConstantInt::get(Type::getInt64Ty(b.getContext()), xork), "");
    return r;
}

// ctor priority window is 111-500 to run strictly after libc runtime ctors (101-110) and strictly before user code (default 65535); a fixed value would be a cross build fingerprint on every protected binary's ctor table
static void EnsureSalt(Module& m)
{
    if (SaltGV) return;
    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);

    SaltGV = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), ".fcpq.salt");
    SaltGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    SaltGV->setAlignment(Align(8));

    FunctionType* ft = FunctionType::get(Type::getVoidTy(c), false);
    Function* saltinit = Function::Create(ft, GlobalValue::InternalLinkage, "__fcpq_salt_init_" + BuildSaltHex, &m);
    saltinit->addFnAttr(Attribute::NoInline);
    BasicBlock* entry = BasicBlock::Create(c, "Entry", saltinit);
    IRBuilder<> ib(entry);
    Function* rdtsc = Intrinsic::getDeclaration(&m, Intrinsic::readcyclecounter);
    Value* tsc = ib.CreateCall(rdtsc, {}, "");
    StoreInst* st = ib.CreateStore(tsc, SaltGV);
    st->setVolatile(true);
    st->setAlignment(Align(8));
    ib.CreateRetVoid();

    const unsigned prio = 111u + (unsigned)(Rng() % 390u);
    appendToGlobalCtors(m, saltinit, (int)prio);
}

static Value* InsertSaltLoad(IRBuilder<>& b)
{
    EnsureSalt(*b.GetInsertBlock()->getParent()->getParent());
    LoadInst* ld = b.CreateLoad(Type::getInt64Ty(b.getContext()), SaltGV, true, "");
    ld->setAlignment(Align(8));
    return ld;
}

static Value* HarvestAuxValue(IRBuilder<>& b, Function& f, DominatorTree& dt)
{
    LLVMContext& c = f.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);
    BasicBlock* targetbb = b.GetInsertBlock();
    BasicBlock::iterator insertit = b.GetInsertPoint();
    if (!targetbb) return nullptr;

    SmallVector<Value*, 16> pool;

    for (Argument& a : f.args())
        if (a.getType()->isIntegerTy(64))
            pool.push_back(&a);

    for (BasicBlock& bb : f)
    {
        if (IsPolluteBlock(&bb)) continue;
        if (IsFcpqInterior(&bb)) continue;
        if (&bb != targetbb && !dt.getNode(&bb)) continue;

        bool domok;
        if (&bb == targetbb) domok = true;
        else domok = dt.dominates(&bb, targetbb);
        if (!domok) continue;

        for (Instruction& i : bb)
        {
            if (&bb == targetbb && BasicBlock::iterator(&i) == insertit) break;
            if (!i.getType()->isIntegerTy(64)) continue;
            if (isa<PHINode>(&i)) continue;
            pool.push_back(&i);
            if (pool.size() >= 32) break;
        }
        if (pool.size() >= 32) break;
    }

    IRBuilder<> loadb(targetbb, insertit);
    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        if (!a->getAllocatedType()->isIntegerTy(64)) continue;
        LoadInst* ld = loadb.CreateLoad(i64, a, true, "");
        ld->setAlignment(Align(8));
        pool.push_back(ld);
        if (pool.size() >= 32) break;
    }

    Value* acc = nullptr;
    if (pool.empty())
    {
        acc = InsertBarrierLoad(b, (unsigned)(Rng() % BarrierPool.size()));
    }
    else
    {
        acc = pool[Rng() % pool.size()];
        const unsigned want = std::min<unsigned>(AuxMixCount, (unsigned)pool.size());
        for (unsigned i = 1; i < want; ++i)
        {
            Value* v = pool[Rng() % pool.size()];
            acc = b.CreateXor(acc, v, "");
        }
    }

    Value* salt = InsertSaltLoad(b);
    acc = b.CreateXor(acc, salt, "");
    return acc;
}

// at every recursion level the SE forks on cached != 0 and memo state is symbolic across the SE path forest so memoization keeps runtime O(n) without reducing attacker 2^n fork cost
static void EnsureFibHelper(Module& m)
{
    if (FibHelper) return;

    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);

    ArrayType* arrty = ArrayType::get(i64, FibMemoSize);
    const std::string memoname = "__fcpq_fib_memo_" + BuildSaltHex;
    FibMemo = new GlobalVariable(m, arrty, false, GlobalValue::InternalLinkage, ConstantAggregateZero::get(arrty), memoname);
    FibMemo->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    FibMemo->setAlignment(Align(8));
    appendToCompilerUsed(m, {FibMemo});

    FunctionType* ft = FunctionType::get(i64, {i64}, false);
    const std::string helpername = "__fcpq_fib_" + BuildSaltHex;
    FibHelper = Function::Create(ft, GlobalValue::InternalLinkage, helpername, &m);

    Argument* narg = &*FibHelper->arg_begin();
    narg->setName("n");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", FibHelper);
    BasicBlock* basecase = BasicBlock::Create(c, "Basecase", FibHelper);
    BasicBlock* rec = BasicBlock::Create(c, "Rec", FibHelper);
    BasicBlock* hitpath = BasicBlock::Create(c, "Hitpath", FibHelper);
    BasicBlock* misspath = BasicBlock::Create(c, "Misspath", FibHelper);

    {
        IRBuilder<> b(entry);
        Value* isbase = b.CreateICmpSLT(narg, ConstantInt::get(i64, 2), "");
        b.CreateCondBr(isbase, basecase, rec);
    }
    {
        IRBuilder<> b(basecase);
        b.CreateRet(narg);
    }

    Value* cached = nullptr;
    {
        IRBuilder<> b(rec);
        Value* idx = b.CreateAnd(narg, ConstantInt::get(i64, FibMemoSize - 1), "");
        Value* slotptr = b.CreateInBoundsGEP(arrty, FibMemo, {ConstantInt::get(i64, 0), idx}, "");
        LoadInst* ld = b.CreateLoad(i64, slotptr, true, "");
        ld->setAlignment(Align(8));
        cached = ld;
        Value* hit = b.CreateICmpNE(cached, ConstantInt::get(i64, 0), "");
        b.CreateCondBr(hit, hitpath, misspath);
    }
    {
        IRBuilder<> b(hitpath);
        b.CreateRet(cached);
    }
    {
        IRBuilder<> b(misspath);
        Value* idx = b.CreateAnd(narg, ConstantInt::get(i64, FibMemoSize - 1), "");
        Value* slot2 = b.CreateInBoundsGEP(arrty, FibMemo, {ConstantInt::get(i64, 0), idx}, "");
        Value* nm1 = b.CreateSub(narg, ConstantInt::get(i64, 1), "");
        Value* nm2 = b.CreateSub(narg, ConstantInt::get(i64, 2), "");
        CallInst* r1 = b.CreateCall(FibHelper, {nm1}, "");
        CallInst* r2 = b.CreateCall(FibHelper, {nm2}, "");
        r1->setTailCall(false);
        r2->setTailCall(false);
        Value* sum = b.CreateAdd(r1, r2, "");
        StoreInst* st = b.CreateStore(sum, slot2);
        st->setVolatile(true);
        st->setAlignment(Align(8));
        b.CreateRet(sum);
    }

    appendToCompilerUsed(m, {FibHelper});
}

// SE cannot unify path context between the fib fork and the lucas fork so every predicate mixing both helpers doubles the fork surface of a pure fib predicate
static void EnsureLucasHelper(Module& m)
{
    if (LucasHelper) return;

    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);

    ArrayType* arrty = ArrayType::get(i64, LucasMemoSize);
    const std::string memoname = "__fcpq_lucas_memo_" + BuildSaltHex;
    LucasMemo = new GlobalVariable(m, arrty, false, GlobalValue::InternalLinkage, ConstantAggregateZero::get(arrty), memoname);
    LucasMemo->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    LucasMemo->setAlignment(Align(8));
    appendToCompilerUsed(m, {LucasMemo});

    FunctionType* ft = FunctionType::get(i64, {i64}, false);
    const std::string helpername = "__fcpq_lucas_" + BuildSaltHex;
    LucasHelper = Function::Create(ft, GlobalValue::InternalLinkage, helpername, &m);

    Argument* narg = &*LucasHelper->arg_begin();
    narg->setName("n");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", LucasHelper);
    BasicBlock* basecase = BasicBlock::Create(c, "Basecase", LucasHelper);
    BasicBlock* rec = BasicBlock::Create(c, "Rec", LucasHelper);
    BasicBlock* hitpath = BasicBlock::Create(c, "Hitpath", LucasHelper);
    BasicBlock* misspath = BasicBlock::Create(c, "Misspath", LucasHelper);

    {
        IRBuilder<> b(entry);
        Value* isbase = b.CreateICmpSLT(narg, ConstantInt::get(i64, 2), "");
        b.CreateCondBr(isbase, basecase, rec);
    }
    {
        IRBuilder<> b(basecase);
        Value* iszero = b.CreateICmpEQ(narg, ConstantInt::get(i64, 0), "");
        Value* r = b.CreateSelect(iszero, ConstantInt::get(i64, 2), ConstantInt::get(i64, 1), "");
        b.CreateRet(r);
    }

    Value* cached = nullptr;
    {
        IRBuilder<> b(rec);
        Value* idx = b.CreateAnd(narg, ConstantInt::get(i64, LucasMemoSize - 1), "");
        Value* slot = b.CreateInBoundsGEP(arrty, LucasMemo, {ConstantInt::get(i64, 0), idx}, "");
        LoadInst* ld = b.CreateLoad(i64, slot, true, "");
        ld->setAlignment(Align(8));
        cached = ld;
        Value* hit = b.CreateICmpNE(cached, ConstantInt::get(i64, 0), "");
        b.CreateCondBr(hit, hitpath, misspath);
    }
    {
        IRBuilder<> b(hitpath);
        b.CreateRet(cached);
    }
    {
        IRBuilder<> b(misspath);
        Value* idx = b.CreateAnd(narg, ConstantInt::get(i64, LucasMemoSize - 1), "");
        Value* slot = b.CreateInBoundsGEP(arrty, LucasMemo, {ConstantInt::get(i64, 0), idx}, "");
        Value* nm1 = b.CreateSub(narg, ConstantInt::get(i64, 1), "");
        Value* nm2 = b.CreateSub(narg, ConstantInt::get(i64, 2), "");
        CallInst* r1 = b.CreateCall(LucasHelper, {nm1}, "");
        CallInst* r2 = b.CreateCall(LucasHelper, {nm2}, "");
        r1->setTailCall(false);
        r2->setTailCall(false);
        Value* sum = b.CreateAdd(r1, r2, "");
        StoreInst* st = b.CreateStore(sum, slot);
        st->setVolatile(true);
        st->setAlignment(Align(8));
        b.CreateRet(sum);
    }

    appendToCompilerUsed(m, {LucasHelper});
}

enum class FibPredKind
{
    LinearRecurrence,
    Cassini,
    Doubling,
    SumSquares,
    Honsberger,
    Pisano,
    LucasCross,
    FibLucasProduct,
    Catalan,
    DOcagne,
};

static FibPredKind PickFibKind(std::mt19937_64& rng)
{
    const unsigned roll = (unsigned)(rng() % 100u);
    if (roll < 18) return FibPredKind::LinearRecurrence;
    if (roll < 33) return FibPredKind::Pisano;
    if (roll < 45) return FibPredKind::Cassini;
    if (roll < 57) return FibPredKind::Doubling;
    if (roll < 67) return FibPredKind::SumSquares;
    if (roll < 77) return FibPredKind::LucasCross;
    if (roll < 85) return FibPredKind::FibLucasProduct;
    if (roll < 92) return FibPredKind::Honsberger;
    if (roll < 97) return FibPredKind::Catalan;
    return FibPredKind::DOcagne;
}

static Value* EmitFibCall(IRBuilder<>& b, Value* arg, const char* name)
{
    CallInst* c = b.CreateCall(FibHelper, {arg}, name);
    c->setTailCall(false);
    return c;
}

static Value* EmitLucasCall(IRBuilder<>& b, Value* arg, const char* name)
{
    CallInst* c = b.CreateCall(LucasHelper, {arg}, name);
    c->setTailCall(false);
    return c;
}

static Value* EmitFibIdentityZero(IRBuilder<>& b, Function& f, DominatorTree& dt, FibPredKind kind)
{
    EnsureFibHelper(*f.getParent());
    EnsureLucasHelper(*f.getParent());
    LLVMContext& c = f.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);

    Value* aux = HarvestAuxValue(b, f, dt);

    switch (kind)
    {
        case FibPredKind::LinearRecurrence:
        {
            Value* n = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x7), ""), ConstantInt::get(i64, 18), "");
            Value* fn = EmitFibCall(b, n, "");
            Value* fnm1 = EmitFibCall(b, b.CreateSub(n, ConstantInt::get(i64, 1), ""), "");
            Value* fnm2 = EmitFibCall(b, b.CreateSub(n, ConstantInt::get(i64, 2), ""), "");
            Value* diff = b.CreateSub(fn, fnm1, "");
            return b.CreateSub(diff, fnm2, "");
        }

        case FibPredKind::Cassini:
        {
            Value* n = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x5), ""), ConstantInt::get(i64, 18), "");
            Value* fnm1 = EmitFibCall(b, b.CreateSub(n, ConstantInt::get(i64, 1), ""), "");
            Value* fn = EmitFibCall(b, n, "");
            Value* fnp1 = EmitFibCall(b, b.CreateAdd(n, ConstantInt::get(i64, 1), ""), "");
            Value* lhsp = b.CreateMul(fnm1, fnp1, "");
            Value* rhsp = b.CreateMul(fn, fn, "");
            Value* x = b.CreateSub(lhsp, rhsp, "");
            Value* xm1 = b.CreateSub(x, ConstantInt::get(i64, 1), "");
            Value* xp1 = b.CreateAdd(x, ConstantInt::get(i64, 1), "");
            return b.CreateMul(xm1, xp1, "");
        }

        case FibPredKind::Doubling:
        {
            Value* n = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x5), ""), ConstantInt::get(i64, 10), "");
            Value* fn = EmitFibCall(b, n, "");
            Value* fnp1 = EmitFibCall(b, b.CreateAdd(n, ConstantInt::get(i64, 1), ""), "");
            Value* f2n = EmitFibCall(b, b.CreateShl(n, ConstantInt::get(i64, 1), ""), "");
            Value* twofnp1 = b.CreateShl(fnp1, ConstantInt::get(i64, 1), "");
            Value* inner = b.CreateSub(twofnp1, fn, "");
            Value* rhs = b.CreateMul(fn, inner, "");
            return b.CreateSub(f2n, rhs, "");
        }

        case FibPredKind::SumSquares:
        {
            Value* masked = b.CreateAnd(aux, ConstantInt::get(i64, 0x3), "");
            Value* mod = b.CreateURem(masked, ConstantInt::get(i64, 3), "");
            Value* n = b.CreateAdd(mod, ConstantInt::get(i64, 10), "");
            Value* np1 = b.CreateAdd(n, ConstantInt::get(i64, 1), "");
            Value* n2p1 = b.CreateAdd(b.CreateShl(n, ConstantInt::get(i64, 1), ""), ConstantInt::get(i64, 1), "");
            Value* fn = EmitFibCall(b, n, "");
            Value* fnp1 = EmitFibCall(b, np1, "");
            Value* f2np1 = EmitFibCall(b, n2p1, "");
            Value* fnsq = b.CreateMul(fn, fn, "");
            Value* fnp1sq = b.CreateMul(fnp1, fnp1, "");
            Value* sum = b.CreateAdd(fnsq, fnp1sq, "");
            return b.CreateSub(f2np1, sum, "");
        }

        case FibPredKind::Honsberger:
        {
            Value* aux2 = HarvestAuxValue(b, f, dt);
            Value* m = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x3), ""), ConstantInt::get(i64, 8), "");
            Value* n = b.CreateAdd(b.CreateAnd(aux2, ConstantInt::get(i64, 0x3), ""), ConstantInt::get(i64, 8), "");
            Value* mn = b.CreateAdd(m, n, "");
            Value* mm1 = b.CreateSub(m, ConstantInt::get(i64, 1), "");
            Value* np1 = b.CreateAdd(n, ConstantInt::get(i64, 1), "");
            Value* fmn = EmitFibCall(b, mn, "");
            Value* fm = EmitFibCall(b, m, "");
            Value* fnp1 = EmitFibCall(b, np1, "");
            Value* fmm1 = EmitFibCall(b, mm1, "");
            Value* fn = EmitFibCall(b, n, "");
            Value* t1 = b.CreateMul(fm, fnp1, "");
            Value* t2 = b.CreateMul(fmm1, fn, "");
            Value* rhs = b.CreateAdd(t1, t2, "");
            return b.CreateSub(fmn, rhs, "");
        }

        case FibPredKind::Pisano:
        {
            const unsigned active = PisanoActive.empty() ? 0u : PisanoActive[(unsigned)(Rng() % PisanoActive.size())];
            const PisanoEntry pe = PisanoPool[active];
            Value* n = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x7), ""), ConstantInt::get(i64, 8), "");
            Value* npi = b.CreateAdd(n, ConstantInt::get(i64, pe.Pi), "");
            Value* fn = EmitFibCall(b, n, "");
            Value* fnp = EmitFibCall(b, npi, "");
            Value* a = b.CreateURem(fn, ConstantInt::get(i64, pe.M), "");
            Value* b2 = b.CreateURem(fnp, ConstantInt::get(i64, pe.M), "");
            return b.CreateSub(a, b2, "");
        }

        case FibPredKind::LucasCross:
        {
            Value* n = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x7), ""), ConstantInt::get(i64, 8), "");
            Value* nm1 = b.CreateSub(n, ConstantInt::get(i64, 1), "");
            Value* np1 = b.CreateAdd(n, ConstantInt::get(i64, 1), "");
            Value* ln = EmitLucasCall(b, n, "");
            Value* fnm1 = EmitFibCall(b, nm1, "");
            Value* fnp1 = EmitFibCall(b, np1, "");
            Value* sum = b.CreateAdd(fnm1, fnp1, "");
            return b.CreateSub(ln, sum, "");
        }

        case FibPredKind::FibLucasProduct:
        {
            Value* masked = b.CreateAnd(aux, ConstantInt::get(i64, 0x5), "");
            Value* n = b.CreateAdd(masked, ConstantInt::get(i64, 10), "");
            Value* n2 = b.CreateShl(n, ConstantInt::get(i64, 1), "");
            Value* f2n = EmitFibCall(b, n2, "");
            Value* fn = EmitFibCall(b, n, "");
            Value* ln = EmitLucasCall(b, n, "");
            Value* prod = b.CreateMul(fn, ln, "");
            return b.CreateSub(f2n, prod, "");
        }

        // catalan identity has a sign ambiguity but (x-y)*(x+y) wraps to 0 in u64 regardless of which sign holds so no runtime parity select is needed
        case FibPredKind::Catalan:
        {
            Value* aux2 = HarvestAuxValue(b, f, dt);
            Value* n = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x3), ""), ConstantInt::get(i64, 10), "");
            Value* r = b.CreateAdd(b.CreateAnd(aux2, ConstantInt::get(i64, 0x3), ""), ConstantInt::get(i64, 2), "");
            Value* nmr = b.CreateSub(n, r, "");
            Value* npr = b.CreateAdd(n, r, "");
            Value* fn = EmitFibCall(b, n, "");
            Value* fnmr = EmitFibCall(b, nmr, "");
            Value* fnpr = EmitFibCall(b, npr, "");
            Value* fr = EmitFibCall(b, r, "");
            Value* fnsq = b.CreateMul(fn, fn, "");
            Value* fnmrfnpr = b.CreateMul(fnmr, fnpr, "");
            Value* x = b.CreateSub(fnsq, fnmrfnpr, "");
            Value* frsq = b.CreateMul(fr, fr, "");
            Value* xmy = b.CreateSub(x, frsq, "");
            Value* xpy = b.CreateAdd(x, frsq, "");
            return b.CreateMul(xmy, xpy, "");
        }

        case FibPredKind::DOcagne:
        {
            Value* aux2 = HarvestAuxValue(b, f, dt);
            Value* m = b.CreateAdd(b.CreateAnd(aux, ConstantInt::get(i64, 0x3), ""), ConstantInt::get(i64, 10), "");
            Value* n = b.CreateAdd(b.CreateAnd(aux2, ConstantInt::get(i64, 0x3), ""), ConstantInt::get(i64, 3), "");
            Value* mp1 = b.CreateAdd(m, ConstantInt::get(i64, 1), "");
            Value* np1 = b.CreateAdd(n, ConstantInt::get(i64, 1), "");
            Value* mmn = b.CreateSub(m, n, "");
            Value* fm = EmitFibCall(b, m, "");
            Value* fnp1 = EmitFibCall(b, np1, "");
            Value* fmp1 = EmitFibCall(b, mp1, "");
            Value* fn = EmitFibCall(b, n, "");
            Value* fmn = EmitFibCall(b, mmn, "");
            Value* lhsp = b.CreateMul(fm, fnp1, "");
            Value* rhsp = b.CreateMul(fmp1, fn, "");
            Value* x = b.CreateSub(lhsp, rhsp, "");
            Value* xmy = b.CreateSub(x, fmn, "");
            Value* xpy = b.CreateAdd(x, fmn, "");
            return b.CreateMul(xmy, xpy, "");
        }
    }
    return ConstantInt::get(i64, 0);
}

enum class CollatzPredKind
{
    Termination,
    MaxValue,
    StepBound,
    OrbitSum,
    OddStepCount,
};

static CollatzPredKind PickCollatzKind(std::mt19937_64& rng)
{
    const unsigned roll = (unsigned)(rng() % 100u);
    if (roll < 35) return CollatzPredKind::Termination;
    if (roll < 60) return CollatzPredKind::MaxValue;
    if (roll < 76) return CollatzPredKind::StepBound;
    if (roll < 89) return CollatzPredKind::OrbitSum;
    return CollatzPredKind::OddStepCount;
}

static Value* EmitCollatzPredicate(IRBuilder<>& b, Function& f, DominatorTree& dt, Instruction* insertptaftersplit, BasicBlock* headbb, BasicBlock* realbb, BasicBlock* pollutebb, bool& outrealontrue)
{
    (void)insertptaftersplit;

    LLVMContext& c = f.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);

    const CollatzPredKind kind = PickCollatzKind(Rng);
    const bool usene = (Rng() & 1u) != 0u;

    const uint64_t sitemask = (Rng() & CollatzInputMask) | 0x3FFULL;

    Value* aux = HarvestAuxValue(b, f, dt);
    Value* masked = b.CreateAnd(aux, ConstantInt::get(i64, sitemask), "");
    Value* n0 = b.CreateOr(masked, ConstantInt::get(i64, 1), "");
    Value* cap = InsertFoldBarrier(b, CollatzSafetyCap);

    BasicBlock* loopbb = BasicBlock::Create(c, "Loop", &f, realbb);
    BasicBlock* bodybb = BasicBlock::Create(c, "Body", &f, realbb);
    BasicBlock* stepbb = BasicBlock::Create(c, "Step", &f, realbb);
    BasicBlock* exitbb = BasicBlock::Create(c, "Exit", &f, realbb);

    b.CreateBr(loopbb);

    IRBuilder<> lb(loopbb);
    PHINode* nphi = lb.CreatePHI(i64, 2, "");
    PHINode* cntphi = lb.CreatePHI(i64, 2, "");
    PHINode* maxphi = nullptr;
    PHINode* sumphi = nullptr;
    PHINode* oddphi = nullptr;
    if (kind == CollatzPredKind::MaxValue) maxphi = lb.CreatePHI(i64, 2, "");
    if (kind == CollatzPredKind::OrbitSum) sumphi = lb.CreatePHI(i64, 2, "");
    if (kind == CollatzPredKind::OddStepCount) oddphi = lb.CreatePHI(i64, 2, "");

    nphi->addIncoming(n0, headbb);
    cntphi->addIncoming(ConstantInt::get(i64, 0), headbb);
    if (maxphi) maxphi->addIncoming(n0, headbb);
    if (sumphi) sumphi->addIncoming(n0, headbb);
    if (oddphi) oddphi->addIncoming(ConstantInt::get(i64, 0), headbb);

    Value* done = lb.CreateICmpEQ(nphi, ConstantInt::get(i64, 1), "");
    lb.CreateCondBr(done, exitbb, bodybb);

    IRBuilder<> bb2(bodybb);
    Value* cntnext = bb2.CreateAdd(cntphi, ConstantInt::get(i64, 1), "");
    Value* overcap = bb2.CreateICmpUGE(cntnext, cap, "");
    bb2.CreateCondBr(overcap, exitbb, stepbb);

    IRBuilder<> sb(stepbb);
    Value* odd = sb.CreateAnd(nphi, ConstantInt::get(i64, 1), "");
    Value* isodd = sb.CreateICmpNE(odd, ConstantInt::get(i64, 0), "");
    Value* n3 = sb.CreateMul(nphi, ConstantInt::get(i64, 3), "");
    Value* n3p1 = sb.CreateAdd(n3, ConstantInt::get(i64, 1), "");
    Value* nd2 = sb.CreateLShr(nphi, ConstantInt::get(i64, 1), "");
    Value* nnext = sb.CreateSelect(isodd, n3p1, nd2, "");
    Value* maxnext = nullptr;
    Value* sumnext = nullptr;
    Value* oddnext = nullptr;
    if (maxphi)
    {
        Value* gt = sb.CreateICmpUGT(nnext, maxphi, "");
        maxnext = sb.CreateSelect(gt, nnext, maxphi, "");
    }
    if (sumphi)
    {
        sumnext = sb.CreateAdd(sumphi, nnext, "");
    }
    if (oddphi)
    {
        Value* oddinc = sb.CreateSelect(isodd, ConstantInt::get(i64, 1), ConstantInt::get(i64, 0), "");
        oddnext = sb.CreateAdd(oddphi, oddinc, "");
    }
    sb.CreateBr(loopbb);

    nphi->addIncoming(nnext, stepbb);
    cntphi->addIncoming(cntnext, stepbb);
    if (maxphi) maxphi->addIncoming(maxnext, stepbb);
    if (sumphi) sumphi->addIncoming(sumnext, stepbb);
    if (oddphi) oddphi->addIncoming(oddnext, stepbb);

    IRBuilder<> eb(exitbb);
    Value* cond = nullptr;
    switch (kind)
    {
        case CollatzPredKind::Termination:
        {
            Value* a = InsertFoldBarrier(eb, 1);
            Value* b2v = InsertFoldBarrier(eb, 1);
            cond = usene ? eb.CreateICmpNE(a, b2v, "") : eb.CreateICmpEQ(a, b2v, "");
            break;
        }
        case CollatzPredKind::MaxValue:
        {
            cond = usene ? eb.CreateICmpULT(maxphi, n0, "") : eb.CreateICmpUGE(maxphi, n0, "");
            break;
        }
        // loopbb to exitbb carries cntphi <= cap-1 by induction and bodybb to exitbb carries cntnext == cap exactly so the merged exitcnt is strictly below cap+1
        case CollatzPredKind::StepBound:
        {
            PHINode* exitcnt = eb.CreatePHI(i64, 2, "");
            exitcnt->addIncoming(cntphi, loopbb);
            exitcnt->addIncoming(cntnext, bodybb);
            Value* capplus1 = InsertFoldBarrier(eb, CollatzSafetyCap + 1);
            cond = usene ? eb.CreateICmpUGE(exitcnt, capplus1, "") : eb.CreateICmpULT(exitcnt, capplus1, "");
            break;
        }
        // sumphi starts at n0 and every n_next added is strictly positive so sumphi is monotonically none decreasing and bounded below by n0 at every loop header and every exit path
        case CollatzPredKind::OrbitSum:
        {
            cond = usene ? eb.CreateICmpULT(sumphi, n0, "") : eb.CreateICmpUGE(sumphi, n0, "");
            break;
        }
        // oddphi starts at 0 and stepbb increments by (isodd ? 1 : 0) while cntphi increments by 1 unconditionally so by induction oddphi <= cntphi at every loop header and every exit path
        case CollatzPredKind::OddStepCount:
        {
            PHINode* exitodd = eb.CreatePHI(i64, 2, "");
            exitodd->addIncoming(oddphi, loopbb);
            exitodd->addIncoming(oddphi, bodybb);
            PHINode* exitcnt2 = eb.CreatePHI(i64, 2, "");
            exitcnt2->addIncoming(cntphi, loopbb);
            exitcnt2->addIncoming(cntnext, bodybb);
            cond = usene ? eb.CreateICmpUGT(exitodd, exitcnt2, "") : eb.CreateICmpULE(exitodd, exitcnt2, "");
            break;
        }
    }

    outrealontrue = !usene;
    eb.CreateCondBr(cond, outrealontrue ? realbb : pollutebb, outrealontrue ? pollutebb : realbb);

    MarkFcpqInterior(loopbb);
    MarkFcpqInterior(bodybb);
    MarkFcpqInterior(stepbb);
    MarkFcpqInterior(exitbb);

    return nullptr;
}

static bool IsEHOrFuncletInterior(BasicBlock* bb)
{
    if (bb->isEHPad()) return true;
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    if (isa<CatchReturnInst>(term) || isa<CleanupReturnInst>(term) || isa<ResumeInst>(term)) return true;
    for (Instruction& i : *bb)
    {
        if (auto* cb = dyn_cast<CallBase>(&i))
        {
            if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) return true;
        }
    }
    return false;
}

static bool BBAlreadyProcessed(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    return term->getMetadata("fcpq.done") != nullptr;
}

static void MarkBBProcessed(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("fcpq.done", MDNode::get(bb->getContext(), {}));
}

static bool IsPolluteBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return false;
    return term->getMetadata("fcpq.pollute") != nullptr;
}

static void MarkPolluteBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("fcpq.pollute", MDNode::get(bb->getContext(), {}));
}

static bool IsFcpqInterior(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return false;
    return term->getMetadata("fcpq.interior") != nullptr;
}

static void MarkFcpqInterior(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("fcpq.interior", MDNode::get(bb->getContext(), {}));
}

static Instruction* PickInsertionPoint(BasicBlock* bb)
{
    Instruction* first = bb->getFirstNonPHIOrDbgOrLifetime();
    if (!first) return nullptr;
    if ((unsigned)bb->size() < MinBBInsts) return nullptr;

    if (bb == &bb->getParent()->getEntryBlock())
    {
        Instruction* lastalloca = nullptr;
        for (Instruction& i : *bb)
            if (isa<AllocaInst>(&i)) lastalloca = &i;
        if (lastalloca)
        {
            Instruction* after = lastalloca->getNextNode();
            if (!after) return nullptr;
            BasicBlock::iterator seek(first);
            while (seek != bb->end() && &*seek != after) ++seek;
            if (seek != bb->end()) first = &*seek;
            else first = after;
        }
    }

    unsigned splittable = 0;
    for (auto it = BasicBlock::iterator(first); it != bb->end(); ++it)
    {
        if (&*it == bb->getTerminator()) break;
        ++splittable;
    }
    if (splittable == 0) return nullptr;

    const unsigned pick = (unsigned)(Rng() % splittable);
    BasicBlock::iterator it(first);
    for (unsigned i = 0; i < pick; ++i) ++it;

    if (it == bb->end()) return nullptr;
    if (&*it == bb->getTerminator()) return nullptr;
    if (isa<PHINode>(&*it)) return nullptr;
    return &*it;
}

struct PolluteSlot
{
    AllocaInst* Base;
    int FieldIdx;
    Type* BaseTy;
    Type* RawTy;
    bool IsPtr;
    IntegerType* IntTy;
};

static Value* MaterializeSlotPtr(IRBuilder<>& b, const PolluteSlot& s)
{
    if (s.FieldIdx < 0) return s.Base;
    IntegerType* i32 = Type::getInt32Ty(b.getContext());
    return b.CreateInBoundsGEP(s.BaseTy, s.Base, {ConstantInt::get(i32, 0), ConstantInt::get(i32, s.FieldIdx)}, "");
}

static Value* EmitSlotLoad(IRBuilder<>& b, const PolluteSlot& s)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    LoadInst* ld = b.CreateLoad(s.RawTy, ptr, "");
    if (!s.IsPtr) return ld;
    return b.CreatePtrToInt(ld, s.IntTy, "");
}

static void EmitSlotStore(IRBuilder<>& b, const PolluteSlot& s, Value* intval, bool volatilestore)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    Value* v = intval;
    if (s.IsPtr) v = b.CreateIntToPtr(intval, s.RawTy, "");
    StoreInst* st = b.CreateStore(v, ptr);
    if (volatilestore) st->setVolatile(true);
}

static void CollectPolluteSlots(Function& f, IntegerType* i32, IntegerType* i64, SmallVectorImpl<PolluteSlot>& out32, SmallVectorImpl<PolluteSlot>& out64)
{
    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        Type* ty = a->getAllocatedType();

        if (ty->isIntegerTy(32))
        {
            out32.push_back({a, -1, ty, ty, false, i32});
            continue;
        }
        if (ty->isIntegerTy(64))
        {
            out64.push_back({a, -1, ty, ty, false, i64});
            continue;
        }
        if (ty->isPointerTy())
        {
            out64.push_back({a, -1, ty, ty, true, i64});
            continue;
        }

        if (auto* st = dyn_cast<StructType>(ty))
        {
            for (unsigned fi = 0; fi < st->getNumElements(); ++fi)
            {
                Type* ft = st->getElementType(fi);
                if (ft->isIntegerTy(32)) out32.push_back({a, (int)fi, ty, ft, false, i32});
                else if (ft->isIntegerTy(64)) out64.push_back({a, (int)fi, ty, ft, false, i64});
                else if (ft->isPointerTy()) out64.push_back({a, (int)fi, ty, ft, true, i64});
            }
        }
    }
}

// cloning a readonly call runs it once per invocation because exactly one of pollute or real fires at runtime, so side effects stay single copy and pollute never fires on legit runs
static unsigned DiamondCloneRealIntoPollute(BasicBlock* realbb, BasicBlock* pollutebb, std::mt19937_64& rng, unsigned maxclone)
{
    if (maxclone == 0) return 0;

    ValueToValueMapTy vmap;
    unsigned cloned = 0;
    const uint64_t mutnonce = rng();

    for (Instruction& i : *realbb)
    {
        if (cloned >= maxclone) break;
        if (i.isTerminator()) break;

        if (isa<PHINode>(&i)) continue;
        if (isa<AllocaInst>(&i)) continue;
        if (auto* cb = dyn_cast<CallBase>(&i))
        {
            if (!cb->onlyReadsMemory()) continue;
            if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) continue;
            if (isa<InvokeInst>(cb) || isa<CallBrInst>(cb)) continue;
            if (cb->isInlineAsm()) continue;
        }
        if (isa<LandingPadInst>(&i)) continue;
        if (isa<CatchPadInst>(&i)) continue;
        if (isa<CleanupPadInst>(&i)) continue;
        if (i.isAtomic()) continue;
        if (isa<FenceInst>(&i)) continue;

        bool allok = true;
        for (Use& u : i.operands())
        {
            Value* op = u.get();
            auto* oi = dyn_cast<Instruction>(op);
            if (!oi) continue;
            if (vmap.count(op)) continue;
            if (oi->getParent() != realbb && oi->getParent() != pollutebb) continue;
            allok = false;
            break;
        }
        if (!allok) continue;

        Instruction* clone = i.clone();
        clone->setName("");
        clone->insertInto(pollutebb, pollutebb->end());
        RemapInstruction(clone, vmap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

        SmallVector<std::pair<unsigned, MDNode*>, 4> allmd;
        clone->getAllMetadata(allmd);
        for (auto& p : allmd) clone->setMetadata(p.first, nullptr);

        if (auto* bo = dyn_cast<BinaryOperator>(clone))
        {
            if ((rng() & 1u) == 0)
            {
                Instruction::BinaryOps newop = bo->getOpcode();
                switch (bo->getOpcode())
                {
                    case Instruction::Add: newop = Instruction::Sub; break;
                    case Instruction::Sub: newop = Instruction::Add; break;
                    case Instruction::And: newop = Instruction::Or; break;
                    case Instruction::Or: newop = Instruction::And; break;
                    case Instruction::Xor: newop = Instruction::Or; break;
                    case Instruction::Mul: newop = Instruction::Add; break;
                    default: break;
                }
                if (newop != bo->getOpcode())
                {
                    IRBuilder<> bb(bo);
                    Value* nb = bb.CreateBinOp(newop, bo->getOperand(0), bo->getOperand(1));
                    bo->replaceAllUsesWith(nb);
                    vmap[&i] = nb;
                    bo->eraseFromParent();
                    ++cloned;
                    continue;
                }
            }
            for (unsigned op = 0; op < bo->getNumOperands(); ++op)
            {
                if (auto* ci = dyn_cast<ConstantInt>(bo->getOperand(op)))
                {
                    if ((rng() % 100u) < 50u)
                    {
                        bo->setOperand(op, ConstantInt::get(ci->getType(), ci->getZExtValue() ^ mutnonce));
                    }
                }
            }
        }
        else if (auto* si = dyn_cast<StoreInst>(clone))
        {
            if ((rng() % 100u) < 30u)
            {
                Value* v = si->getValueOperand();
                if (auto* iv = dyn_cast<IntegerType>(v->getType()))
                {
                    IRBuilder<> bb(si);
                    Value* nv = bb.CreateXor(v, ConstantInt::get(iv, mutnonce), "");
                    si->setOperand(0, nv);
                }
            }
        }

        vmap[&i] = clone;
        ++cloned;
    }

    return cloned;
}

static void FillPolluteBlock(Function& f, BasicBlock* pollutebb, BasicBlock* realbb, BasicBlock* headbb, std::mt19937_64& rng)
{
    LLVMContext& c = f.getContext();
    IRBuilder<> pollb(pollutebb);
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);

    SmallVector<PolluteSlot, 32> slots32, slots64;
    CollectPolluteSlots(f, i32, i64, slots32, slots64);

    SmallVector<Value*, 16> vals32, vals64;
    for (Argument& a : f.args())
    {
        if (a.getType()->isIntegerTy(32)) vals32.push_back(&a);
        else if (a.getType()->isIntegerTy(64)) vals64.push_back(&a);
    }
    for (Instruction& i : *headbb)
    {
        if (i.getType()->isIntegerTy(32)) vals32.push_back(&i);
        else if (i.getType()->isIntegerTy(64)) vals64.push_back(&i);
    }

    const bool have32 = !slots32.empty();
    const bool have64 = !slots64.empty();

    const unsigned diamondcap = std::min<unsigned>((unsigned)(realbb->size() > 1 ? realbb->size() - 1 : 0), DiamondCloneCap);
    const unsigned diamondcloned = DiamondCloneRealIntoPollute(realbb, pollutebb, rng, diamondcap);

    if (!have32 && !have64)
    {
        pollb.SetInsertPoint(pollutebb);
        (void)InsertBarrierLoad(pollb, (unsigned)(rng() % BarrierPool.size()));
        pollb.CreateBr(realbb);
        return;
    }

    pollb.SetInsertPoint(pollutebb);

    unsigned targetops = (unsigned)realbb->size();
    if (targetops < PolluteFloor) targetops = PolluteFloor;
    if (targetops > PolluteCeiling) targetops = PolluteCeiling;
    if (diamondcloned >= targetops) targetops = 2;
    else targetops -= diamondcloned;

    Value* acc = nullptr;
    IntegerType* accty = nullptr;

    auto pickslot = [&](IntegerType* ity) -> const PolluteSlot&
    {
        auto& slots = (ity == i64) ? slots64 : slots32;
        return slots[rng() % slots.size()];
    };

    for (unsigned n = 0; n < targetops; ++n)
    {
        IntegerType* ity;
        if (have32 && have64) ity = (rng() & 1) ? i64 : i32;
        else ity = have64 ? i64 : i32;

        if (acc && accty != ity) acc = nullptr;

        if (!acc || (rng() % 5) == 0)
        {
            acc = EmitSlotLoad(pollb, pickslot(ity));
            accty = ity;
            continue;
        }

        auto& vals = (ity == i64) ? vals64 : vals32;
        Value* rhs = nullptr;
        const unsigned pick = (unsigned)(rng() % 10u);
        if (pick < 6) rhs = EmitSlotLoad(pollb, pickslot(ity));
        else if (pick < 9 && !vals.empty()) rhs = vals[rng() % vals.size()];
        else rhs = ConstantInt::get(ity, rng());

        switch (rng() % 7)
        {
            case 0: acc = pollb.CreateAdd(acc, rhs, ""); break;
            case 1: acc = pollb.CreateSub(acc, rhs, ""); break;
            case 2: acc = pollb.CreateXor(acc, rhs, ""); break;
            case 3: acc = pollb.CreateAnd(acc, rhs, ""); break;
            case 4: acc = pollb.CreateOr(acc, rhs, ""); break;
            case 5: acc = pollb.CreateMul(acc, rhs, ""); break;
            default:
            {
                const unsigned w = ity->getBitWidth();
                unsigned shamt = (unsigned)(rng() % (w - 1u)) + 1u;
                acc = pollb.CreateShl(acc, ConstantInt::get(ity, shamt), "");
                break;
            }
        }

        if ((rng() % 100) < 15) EmitSlotStore(pollb, pickslot(ity), acc, false);
    }

    if (acc) EmitSlotStore(pollb, pickslot(accty), acc, true);
    pollb.CreateBr(realbb);
}

// only the first instruction of real is ever considered for stealing so there are no intervening memory writes between the original site and the new position in head; the first only limit is what keeps readonly calls safe to move
static unsigned StealIntoHead(BasicBlock* headbb, BasicBlock* realbb, unsigned maxsteal)
{
    unsigned stolen = 0;
    while (stolen < maxsteal)
    {
        Instruction* first = realbb->getFirstNonPHIOrDbgOrLifetime();
        if (!first || first->isTerminator()) break;

        if (auto* cb = dyn_cast<CallBase>(first))
        {
            if (!cb->onlyReadsMemory()) break;
            if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) break;
            if (isa<InvokeInst>(cb) || isa<CallBrInst>(cb)) break;
            if (cb->isInlineAsm()) break;
        }
        if (isa<AllocaInst>(first)) break;
        if (isa<LandingPadInst>(first)) break;
        if (isa<CatchPadInst>(first)) break;
        if (isa<CleanupPadInst>(first)) break;
        if (first->isAtomic()) break;
        if (isa<FenceInst>(first)) break;
        if (first->mayHaveSideEffects()) break;

        bool opsok = true;
        for (Use& op : first->operands())
        {
            Value* v = op.get();
            if (auto* oi = dyn_cast<Instruction>(v))
                if (oi->getParent() == realbb)
                {
                    opsok = false;
                    break;
                }
        }
        if (!opsok) break;

        first->removeFromParent();
        first->insertInto(headbb, headbb->end());
        ++stolen;
    }
    return stolen;
}

static bool IsSafeToClone(const Instruction& i)
{
    if (isa<PHINode>(&i)) return false;
    if (i.isTerminator()) return false;
    if (isa<AllocaInst>(&i)) return false;
    if (isa<LandingPadInst>(&i)) return false;
    if (isa<CatchPadInst>(&i)) return false;
    if (isa<CleanupPadInst>(&i)) return false;
    if (isa<VAArgInst>(&i)) return false;
    if (isa<FenceInst>(&i)) return false;
    if (isa<AtomicRMWInst>(&i)) return false;
    if (isa<AtomicCmpXchgInst>(&i)) return false;
    if (auto* cb = dyn_cast<CallBase>(&i))
    {
        if (!cb->onlyReadsMemory()) return false;
        if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) return false;
        if (isa<InvokeInst>(cb)) return false;
        if (isa<CallBrInst>(cb)) return false;
        if (cb->isInlineAsm()) return false;
        return true;
    }
    if (auto* li = dyn_cast<LoadInst>(&i)) return !li->isVolatile() && !li->isAtomic();
    if (auto* si = dyn_cast<StoreInst>(&i)) return !si->isVolatile() && !si->isAtomic();
    if (isa<BinaryOperator>(&i)) return true;
    if (isa<UnaryOperator>(&i)) return true;
    if (isa<CastInst>(&i)) return true;
    if (isa<GetElementPtrInst>(&i)) return true;
    if (isa<ICmpInst>(&i)) return true;
    if (isa<FCmpInst>(&i)) return true;
    if (isa<SelectInst>(&i)) return true;
    if (isa<ExtractValueInst>(&i)) return true;
    if (isa<InsertValueInst>(&i)) return true;
    return false;
}

static bool FindMovableSuffix(BasicBlock& bb, SmallVectorImpl<Instruction*>& out)
{
    out.clear();
    Instruction* term = bb.getTerminator();
    if (!term) return false;
    DenseSet<Instruction*> inwin;
    for (Instruction* curr = term->getPrevNode(); curr; curr = curr->getPrevNode())
    {
        if (!IsSafeToClone(*curr)) break;
        bool ok = true;
        for (User* u : curr->users())
        {
            Instruction* ui = dyn_cast<Instruction>(u);
            if (!ui || !inwin.contains(ui))
            {
                ok = false;
                break;
            }
        }
        if (!ok) break;
        out.push_back(curr);
        inwin.insert(curr);
    }
    std::reverse(out.begin(), out.end());
    return !out.empty();
}

static void TruncateSuffixRandomly(SmallVectorImpl<Instruction*>& suffix, std::mt19937_64& rng)
{
    if (suffix.size() <= DiamondMinSuffix) return;
    const unsigned fullsize = (unsigned)suffix.size();
    const unsigned span = fullsize - DiamondMinSuffix + 1u;
    const unsigned keep = DiamondMinSuffix + (unsigned)(rng() % span);
    const unsigned drop = fullsize - keep;
    if (drop > 0) suffix.erase(suffix.begin(), suffix.begin() + drop);
}

// pollute's stack slot remap reassigns referenced allocas to same typed alternates so pollute writes to a different slot than real but types still match; semantic preservation holds because pollute is never on the legit execution path
static void BuildStackSlotRemap(Function& f, ArrayRef<Instruction*> suffix, std::mt19937_64& rng, DenseMap<AllocaInst*, Value*>& outremap)
{
    SmallVector<AllocaInst*, 16> allallocas;
    for (BasicBlock& bb : f)
        for (Instruction& i : bb)
            if (auto* a = dyn_cast<AllocaInst>(&i)) allallocas.push_back(a);

    DenseMap<Type*, SmallVector<AllocaInst*, 4>> bytype;
    for (AllocaInst* a : allallocas) bytype[a->getAllocatedType()].push_back(a);

    DenseSet<AllocaInst*> referenced;
    for (Instruction* i : suffix)
        for (Use& u : i->operands())
            if (auto* a = dyn_cast<AllocaInst>(u.get())) referenced.insert(a);

    for (AllocaInst* src : referenced)
    {
        auto it = bytype.find(src->getAllocatedType());
        if (it == bytype.end())
        {
            outremap[src] = src;
            continue;
        }
        SmallVector<AllocaInst*, 4>& group = it->second;
        if (group.size() < 2)
        {
            outremap[src] = src;
            continue;
        }
        AllocaInst* pick = nullptr;
        for (int tries = 0; tries < 8 && !pick; ++tries)
        {
            AllocaInst* cand = group[(size_t)(rng() % group.size())];
            if (cand != src) pick = cand;
        }
        outremap[src] = pick ? (Value*)pick : (Value*)src;
    }
}

static void CloneSuffixWithStackRemap(IRBuilder<>& b, ArrayRef<Instruction*> suffix, const DenseMap<AllocaInst*, Value*>& stackremap, std::mt19937_64& rng)
{
    DenseMap<Value*, Value*> valmap;
    const uint64_t mutnonce = rng();
    for (Instruction* orig : suffix)
    {
        Instruction* cloned = orig->clone();
        cloned->setName("");
        SmallVector<std::pair<unsigned, MDNode*>, 4> allmd;
        cloned->getAllMetadata(allmd);
        for (auto& p : allmd) cloned->setMetadata(p.first, nullptr);

        for (unsigned opidx = 0; opidx < cloned->getNumOperands(); ++opidx)
        {
            Value* op = cloned->getOperand(opidx);
            if (!op) continue;
            auto vmit = valmap.find(op);
            if (vmit != valmap.end())
            {
                cloned->setOperand(opidx, vmit->second);
                continue;
            }
            if (auto* a = dyn_cast<AllocaInst>(op))
            {
                auto rmit = stackremap.find(a);
                if (rmit != stackremap.end() && rmit->second != (Value*)a) cloned->setOperand(opidx, rmit->second);
                continue;
            }
            if (auto* ci = dyn_cast<ConstantInt>(op))
            {
                if (!ci->getType()->isIntegerTy()) continue;
                if (isa<GetElementPtrInst>(cloned)) continue;
                if (isa<SwitchInst>(cloned)) continue;
                const unsigned opc = cloned->getOpcode();
                if ((opc == Instruction::Shl || opc == Instruction::LShr || opc == Instruction::AShr) && opidx == 1u) continue;
                if ((unsigned)(rng() % 100u) < ConstPerturbationPct)
                {
                    cloned->setOperand(opidx, ConstantInt::get(ci->getType(), ci->getZExtValue() ^ mutnonce));
                }
            }
        }
        b.Insert(cloned);
        valmap[orig] = cloned;
    }
}

static void PopulatePolluteDiamond(Function& f, BasicBlock* pollutebb, ArrayRef<Instruction*> suffix, BasicBlock* tail, std::mt19937_64& rng)
{
    if (!suffix.empty())
    {
        DenseMap<AllocaInst*, Value*> remap;
        BuildStackSlotRemap(f, suffix, rng, remap);
        IRBuilder<> pb(pollutebb);
        CloneSuffixWithStackRemap(pb, suffix, remap, rng);
    }

    LLVMContext& c = f.getContext();
    IRBuilder<> pb(pollutebb);
    IntegerType* i64 = Type::getInt64Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);

    SmallVector<PolluteSlot, 32> slots32, slots64;
    CollectPolluteSlots(f, i32, i64, slots32, slots64);
    const bool have32 = !slots32.empty();
    const bool have64 = !slots64.empty();

    if (have32 || have64)
    {
        const unsigned targetops = 12u + (unsigned)(rng() % 17u);
        Value* acc = nullptr;
        IntegerType* accty = nullptr;

        auto pickslot = [&](IntegerType* ity) -> const PolluteSlot&
        {
            auto& slots = (ity == i64) ? slots64 : slots32;
            return slots[rng() % slots.size()];
        };

        for (unsigned n = 0; n < targetops; ++n)
        {
            IntegerType* ity;
            if (have32 && have64) ity = (rng() & 1) ? i64 : i32;
            else ity = have64 ? i64 : i32;

            if (acc && accty != ity) acc = nullptr;
            if (!acc || (rng() % 5) == 0)
            {
                acc = EmitSlotLoad(pb, pickslot(ity));
                accty = ity;
                continue;
            }

            Value* rhs = ((rng() % 10u) < 7u) ? EmitSlotLoad(pb, pickslot(ity)) : (Value*)ConstantInt::get(ity, rng());

            switch (rng() % 7)
            {
                case 0: acc = pb.CreateAdd(acc, rhs, ""); break;
                case 1: acc = pb.CreateSub(acc, rhs, ""); break;
                case 2: acc = pb.CreateXor(acc, rhs, ""); break;
                case 3: acc = pb.CreateAnd(acc, rhs, ""); break;
                case 4: acc = pb.CreateOr(acc, rhs, ""); break;
                case 5: acc = pb.CreateMul(acc, rhs, ""); break;
                default:
                {
                    const unsigned w = ity->getBitWidth();
                    unsigned shamt = (unsigned)(rng() % (w - 1u)) + 1u;
                    acc = pb.CreateShl(acc, ConstantInt::get(ity, shamt), "");
                    break;
                }
            }

            if ((rng() % 100) < 15) EmitSlotStore(pb, pickslot(ity), acc, false);
        }

        if (acc) EmitSlotStore(pb, pickslot(accty), acc, true);
    }

    pb.CreateBr(tail);
    MarkPolluteBlock(pollutebb);
}

// k real position is randomised per site so an attacker SE must solve K distinct identities to prove which check is the real one; none real checks compare against barriered nonces at the same shape of compare so each is an independent proof obligation
static bool TryEmitDiamond(Function& f, BasicBlock* bb, DominatorTree& dt, bool pickfib)
{
    SmallVector<Instruction*, 16> suffix;
    if (!FindMovableSuffix(*bb, suffix)) return false;
    TruncateSuffixRandomly(suffix, Rng);
    if (suffix.size() < DiamondMinSuffix) return false;

    LLVMContext& c = f.getContext();
    Instruction* splitpt = suffix.front();

    BasicBlock* tail = bb->splitBasicBlock(splitpt, "Tail");

    BasicBlock* realbb = BasicBlock::Create(c, "Real", &f, tail);
    for (Instruction* inst : suffix)
    {
        inst->removeFromParent();
        inst->insertInto(realbb, realbb->end());
    }
    IRBuilder<>(realbb).CreateBr(tail);

    bb->getTerminator()->eraseFromParent();

    if (!pickfib)
    {
        BasicBlock* pollutebb = BasicBlock::Create(c, "Pollute", &f, tail);
        PopulatePolluteDiamond(f, pollutebb, suffix, tail, Rng);

        dt.recalculate(f);
        IRBuilder<> builder(bb);
        bool realontrue = true;
        (void)EmitCollatzPredicate(builder, f, dt, splitpt, bb, realbb, pollutebb, realontrue);
        MarkBBProcessed(bb);
        return true;
    }

    const unsigned k = FibPFMin + (unsigned)(Rng() % (FibPFMax - FibPFMin + 1u));
    const unsigned kreal = (unsigned)(Rng() % k);

    SmallVector<BasicBlock*, 5> fakes;
    for (unsigned i = 0; i < k - 1; ++i) fakes.push_back(BasicBlock::Create(c, "Fake", &f, tail));
    for (BasicBlock* fb : fakes) PopulatePolluteDiamond(f, fb, suffix, tail, Rng);

    SmallVector<BasicBlock*, 4> mids;
    for (unsigned i = 0; i + 1 < k; ++i) mids.push_back(BasicBlock::Create(c, "Mid", &f, realbb));

    BasicBlock* current = bb;
    unsigned fakeidx = 0;
    for (unsigned i = 0; i < k; ++i)
    {
        dt.recalculate(f);
        IRBuilder<> builder(current);

        const FibPredKind kind = PickFibKind(Rng);
        Value* x = EmitFibIdentityZero(builder, f, dt, kind);

        uint64_t cmpval;
        BasicBlock* target;
        if (i == kreal)
        {
            cmpval = 0;
            target = realbb;
        }
        else
        {
            do { cmpval = Rng(); } while (cmpval == 0);
            target = fakes[fakeidx++];
        }

        Value* cmpbar = InsertFoldBarrier(builder, cmpval);
        Value* cmp = builder.CreateICmpEQ(x, cmpbar, "");

        BasicBlock* next = (i + 1 < k) ? mids[i] : realbb;
        builder.CreateCondBr(cmp, target, next);
        current = next;
    }

    MarkBBProcessed(bb);
    return true;
}

static bool TryEmitClassic(Function& f, BasicBlock* bb, DominatorTree& dt, bool pickfib)
{
    Instruction* splitat = PickInsertionPoint(bb);
    if (!splitat) return false;

    BasicBlock* realbb = bb->splitBasicBlock(splitat, "Real");
    BasicBlock* pollutebb = BasicBlock::Create(f.getContext(), "Pollute", &f, realbb);
    bb->getTerminator()->eraseFromParent();
    IRBuilder<> builder(bb);

    bool realontrue = true;
    Value* cond = nullptr;

    if (pickfib)
    {
        unsigned stealcap = 0;
        const unsigned rbsize = (unsigned)realbb->size();
        if (rbsize > 5)
        {
            const unsigned headroom = rbsize - 5;
            const unsigned ceiling = std::min(StealIntoHeadCeiling, headroom);
            stealcap = 1u + (unsigned)(Rng() % ceiling);
            const unsigned floor = std::min(4u, ceiling);
            if (stealcap < floor) stealcap = floor;
        }
        if (stealcap > 0) (void)StealIntoHead(bb, realbb, stealcap);

        const unsigned k = FibPFMin + (unsigned)(Rng() % (FibPFMax - FibPFMin + 1u));
        const unsigned kreal = (unsigned)(Rng() % k);

        SmallVector<BasicBlock*, 5> fakes;
        fakes.push_back(pollutebb);
        for (unsigned i = 1; i + 1 < k; ++i) fakes.push_back(BasicBlock::Create(f.getContext(), "Fake", &f, realbb));
        for (BasicBlock* fb : fakes)
        {
            FillPolluteBlock(f, fb, realbb, bb, Rng);
            MarkPolluteBlock(fb);
        }

        SmallVector<BasicBlock*, 4> mids;
        for (unsigned i = 0; i + 1 < k; ++i) mids.push_back(BasicBlock::Create(f.getContext(), "Mid", &f, realbb));

        BasicBlock* current = bb;
        unsigned fakeidx = 0;
        for (unsigned i = 0; i < k; ++i)
        {
            dt.recalculate(f);
            IRBuilder<> chainb(current);
            const FibPredKind kind = PickFibKind(Rng);
            Value* x = EmitFibIdentityZero(chainb, f, dt, kind);

            uint64_t cmpval;
            BasicBlock* target;
            if (i == kreal)
            {
                cmpval = 0;
                target = realbb;
            }
            else
            {
                do { cmpval = Rng(); } while (cmpval == 0);
                target = fakes[fakeidx++];
            }

            Value* cmpbar = InsertFoldBarrier(chainb, cmpval);
            Value* cmp = chainb.CreateICmpEQ(x, cmpbar, "");
            BasicBlock* next = (i + 1 < k) ? mids[i] : realbb;
            chainb.CreateCondBr(cmp, target, next);
            current = next;
        }

        (void)cond;
        (void)realontrue;
    }
    else
    {
        FillPolluteBlock(f, pollutebb, realbb, bb, Rng);
        MarkPolluteBlock(pollutebb);
        unsigned stealcap = 0;
        const unsigned rbsize = (unsigned)realbb->size();
        if (rbsize > 5)
        {
            const unsigned headroom = rbsize - 5;
            const unsigned ceiling = std::min(StealIntoHeadCeiling, headroom);
            stealcap = 1u + (unsigned)(Rng() % ceiling);
            const unsigned floor = std::min(4u, ceiling);
            if (stealcap < floor) stealcap = floor;
        }
        if (stealcap > 0) (void)StealIntoHead(bb, realbb, stealcap);
        dt.recalculate(f);
        builder.SetInsertPoint(bb);
        (void)EmitCollatzPredicate(builder, f, dt, splitat, bb, realbb, pollutebb, realontrue);
    }

    MarkBBProcessed(bb);
    return true;
}

static bool TryEmitAtBB(Function& f, BasicBlock* bb, DominatorTree& dt)
{
    if (BBAlreadyProcessed(bb)) return false;
    if (SkipEntry && bb == &f.getEntryBlock()) return false;
    if (IsEHOrFuncletInterior(bb)) return false;
    if (bb->size() < MinBBInsts) return false;

    const unsigned roll = (unsigned)(Rng() % 100u);
    const bool pickfib = roll < FibWeightPct;

    if (TryEmitDiamond(f, bb, dt, pickfib)) return true;
    if (TryEmitClassic(f, bb, dt, pickfib)) return true;
    return false;
}

static BasicBlock* RealBBFromSplit(BasicBlock* bb)
{
    Instruction* t = bb->getTerminator();
    if (!t) return nullptr;

    if (auto* br = dyn_cast<BranchInst>(t))
    {
        if (br->isConditional())
        {
            for (unsigned i = 0; i < br->getNumSuccessors(); ++i)
            {
                BasicBlock* s = br->getSuccessor(i);
                if (!IsPolluteBlock(s)) return s;
            }
            return nullptr;
        }
        BasicBlock* cur = br->getSuccessor(0);
        SmallPtrSet<BasicBlock*, 8> visited;
        while (cur && IsFcpqInterior(cur))
        {
            if (!visited.insert(cur).second) return nullptr;
            Instruction* ct = cur->getTerminator();
            if (!ct) return nullptr;
            if (auto* cb = dyn_cast<BranchInst>(ct))
            {
                BasicBlock* next = nullptr;
                for (unsigned i = 0; i < cb->getNumSuccessors(); ++i)
                {
                    BasicBlock* s = cb->getSuccessor(i);
                    if (IsPolluteBlock(s)) continue;
                    if (visited.count(s)) continue;
                    next = s;
                    break;
                }
                if (!next) return nullptr;
                cur = next;
            }
            else return nullptr;
        }
        return cur;
    }
    return nullptr;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;

    bool noobf = HasNoObfuscateAnnotation(f);
    bool nofcpq = HasNoFibCollAnnotation(f);
    bool hasobf = HasObfuscateAnnotation(f);
    if (noobf || nofcpq) return false;
    if (!hasobf) return false;

    SmallVector<BasicBlock*, 64> worklist;
    SmallPtrSet<BasicBlock*, 32> seen;
    for (BasicBlock& bb : f)
    {
        worklist.push_back(&bb);
        seen.insert(&bb);
    }

    unsigned emitted = 0;
    bool mutated = false;

    while (!worklist.empty() && emitted < PerFunctionCap)
    {
        BasicBlock* bb = worklist.pop_back_val();
        if (!bb) continue;
        if (bb->getParent() != &f) continue;
        if (IsPolluteBlock(bb)) continue;
        if (IsFcpqInterior(bb)) continue;

        DominatorTree dt;
        dt.recalculate(f);
        LoopInfo li;
        li.analyze(dt);
        if (SkipLoops && li.getLoopFor(bb) != nullptr) continue;

        bool ok = TryEmitAtBB(f, bb, dt);
        if (!ok) continue;

        ++emitted;
        mutated = true;

        if (BasicBlock* realbb = RealBBFromSplit(bb))
            if (seen.insert(realbb).second)
                worklist.push_back(realbb);
    }

    return mutated;
}

bool RunFibCollatzOpaque(Module& m)
{
    if (!FibCollatzOpaqueEnable) return false;

    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    seed ^= 0x465350515F5F5F5FULL;
    Rng.seed(seed);

    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%08x", (unsigned)(Rng() & 0xFFFFFFFFu));
        BuildSaltHex = buf;
    }

    BarrierPool.clear();
    BarrierValues.clear();
    FibHelper = nullptr;
    FibMemo = nullptr;
    LucasHelper = nullptr;
    LucasMemo = nullptr;
    SaltGV = nullptr;

    PisanoActive.clear();
    {
        std::vector<unsigned> allidx;
        allidx.reserve(PisanoPoolSize);
        for (unsigned i = 0; i < PisanoPoolSize; ++i) allidx.push_back(i);
        for (unsigned i = (unsigned)allidx.size(); i > 1; --i)
        {
            const unsigned j = (unsigned)(Rng() % i);
            std::swap(allidx[i - 1], allidx[j]);
        }
        const unsigned span = PisanoActiveMax - PisanoActiveMin + 1u;
        const unsigned numactive = PisanoActiveMin + (unsigned)(Rng() % span);
        for (unsigned i = 0; i < numactive && i < (unsigned)allidx.size(); ++i) PisanoActive.push_back(allidx[i]);
    }

    BuildBarrierPool(m);

    bool mutated = false;
    for (Function& f : m) mutated |= ProcessFunction(f);

    return mutated;
}

PreservedAnalyses FibCollatzOpaquePass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunFibCollatzOpaque(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}