use {
    crate::{
        const_expr::EvaluationContext,
        data::Data,
        elem::Elem,
        extern_val::ExternVal,
        func::Func,
        global::Global,
        guarded::{GuardedBTreeMap, GuardedBTreeMapIter, GuardedBoxSlice, GuardedVec},
        mem::Mem,
        store::{InternedFuncType, StoreGuard},
        table::Table,
    },
    std::{cell::OnceCell, sync::Arc},
};

/// A [`Module`](`crate::Module`) instance.
#[derive(Clone, Debug)]
pub struct Instance {
    inner: Arc<OnceCell<InstanceInner>>,
}

impl Instance {
    /// Returns the [`ExternVal`] of the export with the given name in this [`Instance`], if it exists.
    pub fn exported_val(&self, name: &str) -> Option<ExternVal> {
        self.inner().exports.get(name)
    }

    /// Returns the exported [`Func`] with the given name in this [`Instance`], if it exists.
    pub fn exported_func(&self, name: &str) -> Option<Func> {
        self.exported_val(name).and_then(|val| val.to_func())
    }

    /// Returns the exported [`Table`] with the given name in this [`Instance`], if it exists.
    pub fn exported_table(&self, name: &str) -> Option<Table> {
        self.exported_val(name).and_then(|val| val.to_table())
    }

    /// Returns the exported [`Mem`] with the given name in this [`Instance`], if it exists.
    pub fn exported_mem(&self, name: &str) -> Option<Mem> {
        self.exported_val(name).and_then(|val| val.to_mem())
    }

    /// Returns the exported [`Global`] with the given name in this [`Instance`], if it exists.
    pub fn exported_global(&self, name: &str) -> Option<Global> {
        self.exported_val(name).and_then(|val| val.to_global())
    }

    /// Returns an iterator over the exports in this [`Instance`].
    pub fn exports(&self) -> InstanceExports<'_> {
        InstanceExports {
            iter: self.inner().exports.iter(),
        }
    }

    /// Creates an uninitialized [`Instance`].
    pub(crate) fn uninited() -> Instance {
        Instance {
            inner: Arc::new(OnceCell::new()),
        }
    }

    /// Initialize this [`Instance`] with the given [`InstanceIniter`].
    pub(crate) fn init(&self, initer: InstanceIniter) {
        self.inner
            .set(InstanceInner {
                types: initer.types.into(),
                funcs: initer.funcs.into(),
                tables: initer.tables.into(),
                mems: initer.mems.into(),
                globals: initer.globals.into(),
                elems: initer.elems.into(),
                datas: initer.datas.into(),
                exports: initer.exports,
            })
            .expect("instance already initialized");
    }

    /// Returns the [`InternedFuncType`] at the given index in this [`Instance`], if it exists.
    pub(crate) fn type_(&self, idx: u32) -> Option<InternedFuncType> {
        self.inner().types.get(idx as usize)
    }

    /// Returns the [`Func`] at the given index in this [`Instance`], if it exists.
    pub(crate) fn func(&self, idx: u32) -> Option<Func> {
        self.inner().funcs.get(idx as usize)
    }

    /// Returns the [`Table`] at the given index in this [`Instance`], if it exists.
    pub(crate) fn table(&self, idx: u32) -> Option<Table> {
        self.inner().tables.get(idx as usize)
    }

    /// Returns the [`Mem`] at the given index in this [`Instance`], if it exists.
    pub(crate) fn mem(&self, idx: u32) -> Option<Mem> {
        self.inner().mems.get(idx as usize)
    }

    /// Returns the [`Global`] at the given index in this [`Instance`], if it exists.
    pub(crate) fn global(&self, idx: u32) -> Option<Global> {
        self.inner().globals.get(idx as usize)
    }

    /// Returns the [`Elem`] at the given index in this [`Instance`], if it exists.
    pub(crate) fn elem(&self, idx: u32) -> Option<Elem> {
        self.inner().elems.get(idx as usize)
    }

    /// Returns the [`Data`] at the given index in this [`Instance`], if it exists.
    pub(crate) fn data(&self, idx: u32) -> Option<Data> {
        self.inner().datas.get(idx as usize)
    }

    fn inner(&self) -> &InstanceInner {
        self.inner.get().expect("instance not yet initialized")
    }
}

impl EvaluationContext for Instance {
    fn func(&self, idx: u32) -> Option<Func> {
        self.func(idx)
    }

    fn global(&self, idx: u32) -> Option<Global> {
        self.global(idx)
    }
}

/// An iterator over the exports in an [`Instance`].
pub struct InstanceExports<'a> {
    iter: GuardedBTreeMapIter<'a, Arc<str>, ExternVal>,
}

impl<'a> Iterator for InstanceExports<'a> {
    type Item = (&'a str, ExternVal);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(name, val)| (&**name, val))
    }
}

#[derive(Debug)]
struct InstanceInner {
    types: GuardedBoxSlice<InternedFuncType>,
    funcs: GuardedBoxSlice<Func>,
    tables: GuardedBoxSlice<Table>,
    mems: GuardedBoxSlice<Mem>,
    globals: GuardedBoxSlice<Global>,
    elems: GuardedBoxSlice<Elem>,
    datas: GuardedBoxSlice<Data>,
    exports: GuardedBTreeMap<Arc<str>, ExternVal>,
}

/// An initializer for an [`Instance`].
#[derive(Debug)]
pub(crate) struct InstanceIniter {
    types: GuardedVec<InternedFuncType>,
    funcs: GuardedVec<Func>,
    tables: GuardedVec<Table>,
    mems: GuardedVec<Mem>,
    globals: GuardedVec<Global>,
    elems: GuardedVec<Elem>,
    datas: GuardedVec<Data>,
    exports: GuardedBTreeMap<Arc<str>, ExternVal>,
}

impl InstanceIniter {
    /// Creates a new [`InstanceIniter`].
    pub(crate) fn new(guard: StoreGuard) -> InstanceIniter {
        InstanceIniter {
            types: GuardedVec::new(guard),
            funcs: GuardedVec::new(guard),
            tables: GuardedVec::new(guard),
            mems: GuardedVec::new(guard),
            globals: GuardedVec::new(guard),
            elems: GuardedVec::new(guard),
            datas: GuardedVec::new(guard),
            exports: GuardedBTreeMap::new(guard),
        }
    }

    /// Returns the [`Func`] at the given index in this [`InstanceIniter`], if it exists.
    pub(crate) fn func(&self, idx: u32) -> Option<Func> {
        self.funcs.get(idx as usize)
    }

    /// Returns the [`Table`] at the given index in this [`InstanceIniter`], if it exists.
    pub(crate) fn table(&self, idx: u32) -> Option<Table> {
        self.tables.get(idx as usize)
    }

    /// Returns the [`Mem`] at the given index in this [`InstanceIniter`], if it exists.
    pub(crate) fn mem(&self, idx: u32) -> Option<Mem> {
        self.mems.get(idx as usize)
    }

    /// Returns the [`Global`] at the given index in this [`InstanceIniter`], if it exists.
    pub(crate) fn global(&self, idx: u32) -> Option<Global> {
        self.globals.get(idx as usize)
    }

    /// Appends the given [`InternedFuncType`] to this [`InstanceIniter`].
    pub(crate) fn push_type(&mut self, type_: InternedFuncType) {
        self.types.push(type_)
    }

    /// Appends the given [`Func`] to this [`InstanceIniter`].
    pub(crate) fn push_func(&mut self, func: Func) {
        self.funcs.push(func)
    }

    /// Appends the given [`Table`] to this [`InstanceIniter`].
    pub(crate) fn push_table(&mut self, table: Table) {
        self.tables.push(table)
    }

    /// Appends the given [`Mem`] to this [`InstanceIniter`].
    pub(crate) fn push_mem(&mut self, mem: Mem) {
        self.mems.push(mem)
    }

    /// Appends the given [`Global`] to this [`InstanceIniter`].
    pub(crate) fn push_global(&mut self, global: Global) {
        self.globals.push(global)
    }

    /// Appends the given [`Elem`] to this [`InstanceIniter`].
    pub(crate) fn push_elem(&mut self, elem: Elem) {
        self.elems.push(elem)
    }

    /// Appends the given [`Data`] to this [`InstanceIniter`].
    pub(crate) fn push_data(&mut self, data: Data) {
        self.datas.push(data)
    }

    /// Appends an export with the given name and [`ExternVal`] to this [`InstanceIniter`].
    pub(crate) fn push_export(&mut self, name: Arc<str>, val: ExternVal) {
        self.exports.insert(name, val);
    }
}

impl EvaluationContext for InstanceIniter {
    fn func(&self, idx: u32) -> Option<Func> {
        self.func(idx)
    }

    fn global(&self, idx: u32) -> Option<Global> {
        self.global(idx)
    }
}
