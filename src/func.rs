use {
    crate::{
        code::Code,
        decode::{Decode, DecodeError, Decoder},
        error::Error,
        exec,
        instance::Instance,
        into_func::IntoFunc,
        stack::Stack,
        store::{Handle, InternedFuncType, Store, StoreId, UnguardedHandle},
        val::{Val, ValType},
    },
    std::{error, fmt, mem, sync::Arc},
};

/// A Wasm function.
///
/// A [`Func`] is either a Wasm function or a host function.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Func(pub(crate) Handle<FuncEntity>);

impl Func {
    /// Creates a new host function that wraps the given closure.
    pub fn wrap<Ts, U>(store: &mut Store, f: impl IntoFunc<Ts, U>) -> Self {
        let (type_, trampoline) = f.into_func();
        let type_ = store.get_or_intern_type(&type_);
        Self(store.insert_func(FuncEntity::Host(HostFuncEntity::new(type_, trampoline))))
    }

    /// Returns the [`FuncType`] of this [`Func`].
    pub fn type_(self, store: &Store) -> &FuncType {
        store.resolve_type(self.0.as_ref(store).type_())
    }

    /// Calls this [`Func`] with the given arguments.
    ///
    /// The results are written to the `results` slice.
    ///
    /// # Errors
    ///
    /// - If the argument count does not match the expected parameter count.
    /// - If the actual result count does not match the expected result count.
    /// - If the argument types do not match the expected parameter types.
    pub fn call(self, mut context: impl Context, args: &[Val], results: &mut [Val]) -> Result<(), Error> {
        let (store, stack) = context.into_parts();
        let type_ = self.type_(store);
        if args.len() != type_.params().len() {
            return Err(FuncError::ParamCountMismatch)?;
        }
        if results.len() != type_.results().len() {
            return Err(FuncError::ResultCountMismatch)?;
        }
        for (arg, param_type) in args.iter().zip(type_.params().iter().copied()) {
            if arg.type_() != param_type {
                return Err(FuncError::ParamTypeMismatch)?;
            }
        }
        if let Some(stack) = stack {
            exec::exec(store, stack, self, args, results)
        } else {
            let mut stack = Stack::new(8 * 1024 * 1024);
            exec::exec(store, &mut stack, self, args, results)
        }
    }

    /// Creates a new Wasm function from its raw parts.
    pub(crate) fn new_wasm(
        store: &mut Store,
        type_: InternedFuncType,
        instance: Instance,
        code: UncompiledFuncBody,
    ) -> Self {
        Self(store.insert_func(FuncEntity::Wasm(WasmFuncEntity::new(type_, instance, code))))
    }

    /// Converts the given [`UnguardedFunc`] to a [`Func`].
    ///
    /// # Safety
    ///
    /// The given [`UnguardedFunc`] must be owned by the [`Store`] with the given [`StoreId`].
    pub(crate) unsafe fn from_unguarded(func: UnguardedFunc, store_id: StoreId) -> Self {
        Self(Handle::from_unguarded(func, store_id))
    }

    /// Converts this [`Func`] to an [`UnguardedFunc`].
    ///
    /// # Panics
    ///
    /// This [`Func`] is not owned by the [`Store`] with the given [`StoreId`].
    pub(crate) fn to_unguarded(self, store_id: StoreId) -> UnguardedFunc {
        self.0.to_unguarded(store_id)
    }

    /// Ensures that this [`Func`] is compiled, if it is a Wasm function.
    pub(crate) fn compile(self, store: &mut Store) {
        let FuncEntity::Wasm(func) = self.0.as_mut(store) else {
            return;
        };
        let instance = func.instance().clone();
        let code = match mem::replace(func.code_mut(), FuncBody::Compiling) {
            FuncBody::Uncompiled(code) => {
                let engine = store.engine().clone();
                engine.compile(store, self, &instance, &code)
            }
            FuncBody::Compiling => panic!("function is already being compiled"),
            FuncBody::Compiled(state) => state,
        };
        let FuncEntity::Wasm(func) = self.0.as_mut(store) else {
            unreachable!();
        };
        *func.code_mut() = FuncBody::Compiled(code);
    }
}

pub trait Context {
    fn into_parts(&mut self) -> (&mut Store, Option<&mut Stack>);
}

impl<T> Context for &mut T
where 
    T: Context
{
    fn into_parts(&mut self) -> (&mut Store, Option<&mut Stack>) {
        (*self).into_parts()
    }
}

impl Context for &mut Store {
    fn into_parts(&mut self) -> (&mut Store, Option<&mut Stack>) {
        (self, None)
    }
}

/// An unguarded version of [`Func`].
pub(crate) type UnguardedFunc = UnguardedHandle<FuncEntity>;

/// The type of a [`Func`].
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FuncType {
    params_results: Arc<[ValType]>,
    param_count: usize,
}

impl FuncType {
    /// Creates a new [`FuncType`] with the given parameters and results.
    pub fn new(
        params: impl IntoIterator<Item = ValType>,
        results: impl IntoIterator<Item = ValType>,
    ) -> Self {
        let mut params_results = params.into_iter().collect::<Vec<_>>();
        let param_count = params_results.len();
        params_results.extend(results);
        Self {
            params_results: params_results.into(),
            param_count,
        }
    }

    /// Returns the parameters of this [`FuncType`].
    pub fn params(&self) -> &[ValType] {
        &self.params_results[..self.param_count]
    }

    /// Returns the results of this [`FuncType`].
    pub fn results(&self) -> &[ValType] {
        &self.params_results[self.param_count..]
    }

    /// Creates a [`FuncType`] from an optional [`ValType`], which is a shorthand for the
    /// [`FuncType`] [] -> [`ValType`?].
    pub(crate) fn from_val_type(type_: Option<ValType>) -> FuncType {
        thread_local! {
            static TYPES: [FuncType; 7] = [
                FuncType::new(vec![], vec![]),
                FuncType::new(vec![], vec![ValType::I32]),
                FuncType::new(vec![], vec![ValType::I64]),
                FuncType::new(vec![], vec![ValType::F32]),
                FuncType::new(vec![], vec![ValType::F64]),
                FuncType::new(vec![], vec![ValType::FuncRef]),
                FuncType::new(vec![], vec![ValType::ExternRef]),
            ];
        }

        TYPES.with(|types| match type_ {
            None => types[0].clone(),
            Some(ValType::I32) => types[1].clone(),
            Some(ValType::I64) => types[2].clone(),
            Some(ValType::F32) => types[3].clone(),
            Some(ValType::F64) => types[4].clone(),
            Some(ValType::FuncRef) => types[5].clone(),
            Some(ValType::ExternRef) => types[6].clone(),
        })
    }
}

impl Decode for FuncType {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        if decoder.read_byte()? != 0x60 {
            return Err(DecodeError::new("malformed function type"))?;
        }
        let mut param_result_types: Vec<_> = decoder.decode_iter()?.collect::<Result<_, _>>()?;
        let param_count = param_result_types.len();
        let result_types = decoder.decode_iter()?;
        param_result_types.reserve(result_types.size_hint().0);
        for result_type in result_types {
            param_result_types.push(result_type?);
        }
        Ok(Self {
            params_results: param_result_types.into(),
            param_count,
        })
    }
}

/// An error that can occur when operating on a [`Func`].
#[derive(Clone, Copy, Debug)]
pub enum FuncError {
    ParamCountMismatch,
    ParamTypeMismatch,
    ResultCountMismatch,
}

impl fmt::Display for FuncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParamCountMismatch => write!(f, "function parameter count mismatch"),
            Self::ParamTypeMismatch => write!(f, "function parameter type mismatch"),
            Self::ResultCountMismatch => write!(f, "function result count mismatch"),
        }
    }
}

impl error::Error for FuncError {}

/// The representation of a [`Func`] in a [`Store`].
#[derive(Debug)]
pub enum FuncEntity {
    Wasm(WasmFuncEntity),
    Host(HostFuncEntity),
}

impl FuncEntity {
    /// Returns the [`FuncType`] of this [`FuncEntity`].
    pub(crate) fn type_(&self) -> InternedFuncType {
        match self {
            Self::Wasm(func) => func.type_(),
            Self::Host(func) => func.type_(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct WasmFuncEntity {
    type_: InternedFuncType,
    instance: Instance,
    code: FuncBody,
}

impl WasmFuncEntity {
    /// Creates a new [`WasmFuncEntity`] from its raw parts.
    fn new(type_: InternedFuncType, instance: Instance, code: UncompiledFuncBody) -> WasmFuncEntity {
        WasmFuncEntity {
            type_,
            instance,
            code: FuncBody::Uncompiled(code),
        }
    }

    /// Returns the [`InternedFuncType`] of this [`WasmFuncEntity`].
    pub(crate) fn type_(&self) -> InternedFuncType {
        self.type_
    }

    /// Returns the [`Instance`] of this [`WasmFuncEntity`].
    pub(crate) fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Returns a reference to the [`Code`] of this [`WasmFuncEntity`].
    pub(crate) fn code(&self) -> &FuncBody {
        &self.code
    }

    /// Returns a mutable reference to the [`Code`] of this [`WasmFuncEntity`].
    pub(crate) fn code_mut(&mut self) -> &mut FuncBody {
        &mut self.code
    }
}



#[derive(Debug)]
pub(crate) enum FuncBody {
    Uncompiled(UncompiledFuncBody),
    Compiling,
    Compiled(CompiledFuncBody),
}

#[derive(Clone, Debug)]
pub(crate) struct UncompiledFuncBody {
    pub(crate) locals: Box<[ValType]>,
    pub(crate) expr: Arc<[u8]>,
}

impl Decode for UncompiledFuncBody {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        use std::iter;

        let mut code_decoder = decoder.decode_decoder()?;
        Ok(Self {
            locals: {
                let mut locals = Vec::new();
                for _ in 0u32..code_decoder.decode()? {
                    let count = code_decoder.decode()?;
                    if count > usize::try_from(u32::MAX).unwrap() - locals.len() {
                        return Err(DecodeError::new("too many locals"));
                    }
                    locals.extend(iter::repeat(code_decoder.decode::<ValType>()?).take(count));
                }
                locals.into()
            },
            expr: code_decoder.read_bytes_until_end().into(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct CompiledFuncBody {
    pub(crate) max_stack_height: usize,
    pub(crate) local_count: usize,
    pub(crate) code: Code,
}

pub(crate) type InstrSlot = usize;

#[derive(Debug)]
pub struct HostFuncEntity {
    type_: InternedFuncType,
    trampoline: HostFuncTrampoline,
}

impl HostFuncEntity {
    /// Creates a new [`HostFuncEntity`] from its raw parts.
    pub(crate) fn new(type_: InternedFuncType, trampoline: HostFuncTrampoline) -> Self {
        Self { type_, trampoline }
    }

    /// Returns the [`InternedFuncType`] of this [`HostFuncEntity`].
    pub(crate) fn type_(&self) -> InternedFuncType {
        self.type_
    }

    /// Returns the [`HostFuncTrampoline`] of this [`HostFuncEntity`].
    pub(crate) fn trampoline(&self) -> &HostFuncTrampoline {
        &self.trampoline
    }
}

#[derive(Clone)]
pub struct HostFuncTrampoline {
    inner: Arc<dyn Fn(Caller) -> Result<(), Error> + Send + Sync + 'static>,
}

impl HostFuncTrampoline {
    pub fn new(
        inner: impl Fn(Caller) -> Result<(), Error> + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub(crate) fn call(&self, caller: Caller) -> Result<(), Error> {
        (self.inner)(caller)
    }
}

impl fmt::Debug for HostFuncTrampoline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("HostFuncTrampoline").finish()
    }
}

#[derive(Debug)]
pub struct Caller<'a> {
    pub(crate) store: &'a mut Store,
    pub(crate) stack: &'a mut Stack,
}