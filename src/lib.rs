mod aliasable_box;
mod cast;
mod instr;
mod code;
mod compile;
mod config;
mod const_expr;
mod data;
mod decode;
mod downcast;
mod elem;
mod engine;
mod error;
mod exec;
mod extern_;
mod extern_val;
mod func;
mod global;
mod guarded;
mod instance;
mod into_func;
mod limits;
mod linker;
mod mem;
mod module;
mod ops;
mod ref_;
mod stack;
mod store;
mod table;
mod trap;
mod val;
mod validate;

pub use self::{
    decode::DecodeError,
    engine::Engine,
    error::Error,
    extern_::Extern,
    extern_val::{ExternType, ExternVal},
    func::{Func, FuncError, FuncType},
    global::{Global, GlobalError, GlobalType, Mut},
    instance::{Instance, InstanceExports},
    limits::Limits,
    linker::{InstantiateError, Linker},
    mem::{Mem, MemError, MemType},
    module::{Module, ModuleExports, ModuleImports},
    ref_::{ExternRef, FuncRef, Ref, RefType},
    store::Store,
    table::{Table, TableError, TableType},
    val::{Val, ValType},
};
