mod aliasable;
mod cast;
mod instr;
mod code;
mod compiler;
mod config;
mod const_expr;
mod data;
mod decode;
mod downcast;
mod elem;
mod engine;
mod error;
mod executor;
mod extern_;
mod extern_val;
mod func;
mod guarded;
mod instance;
mod into_func;
mod limits;
mod linker;
mod mem;
mod module;
mod ops;
mod ref_;
mod runtime;
mod stack;
mod store;
mod trap;
mod val;
mod validator;

pub use self::{
    decode::DecodeError,
    engine::Engine,
    error::Error,
    extern_::Extern,
    extern_val::{ExternType, ExternVal},
    func::{Func, FuncError, FuncType},
    runtime::{
        global::{Global, GlobalError, GlobalType, Mutability},
        table::{Table, TableError, TableType},
    },
    instance::{Instance, InstanceExports},
    limits::Limits,
    linker::{InstantiateError, Linker},
    mem::{Mem, MemError, MemType},
    module::{Module, ModuleExports, ModuleImports},
    ref_::{ExternRef, FuncRef, Ref, RefType},
    store::Store,
    val::{Val, ValType},
};
