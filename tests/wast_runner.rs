use {
    makepad_stitch::{
        Engine, Error, Extern, ExternRef, Func, FuncRef, Global, GlobalType, Instance, Limits,
        Linker, Mem, MemType, Module, Mutability, Ref, RefType, Store, Table, TableType, Val,
        ValType,
    },
    std::{collections::HashMap, sync::Arc},
    wast::{
        core::{HeapType, NanPattern, WastArgCore, WastRetCore},
        parser,
        parser::ParseBuffer,
        QuoteWat, Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet, Wat,
    },
};

#[derive(Debug)]
pub struct WastRunner {
    store: Store,
    linker: Linker,
    instances_by_name: HashMap<String, Instance>,
    current_instance: Option<Instance>,
}

impl WastRunner {
    pub fn new() -> Self {
        let mut store = Store::new(Engine::new());
        let mut linker = Linker::new();
        let print = Func::wrap(&mut store, || {
            println!("print");
        });
        let print_i32 = Func::wrap(&mut store, |val: i32| {
            println!("{}", val);
        });
        let print_i64 = Func::wrap(&mut store, |val: i64| {
            println!("{}", val);
        });
        let print_f32 = Func::wrap(&mut store, |val: f32| {
            println!("{}", val);
        });
        let print_f64 = Func::wrap(&mut store, |val: f64| {
            println!("{}", val);
        });
        let print_i32_f32 = Func::wrap(&mut store, |val_0: i32, val_1: f32| {
            println!("{} {}", val_0, val_1);
        });
        let print_f64_f64 = Func::wrap(&mut store, |val_0: f64, val_1: f64| {
            println!("{} {}", val_0, val_1);
        });
        let table = Table::new(
            &mut store,
            TableType::new(
                RefType::FuncRef,
                10,
                Some(20),
            ),
            Ref::null(RefType::FuncRef),
        )
        .unwrap();
        let memory = Mem::new(
            &mut store,
            MemType {
                limits: Limits {
                    min: 1,
                    max: Some(2),
                },
            },
        );
        let global_i32 = Global::new(
            &mut store,
            GlobalType::new(ValType::I32, Mutability::Const),
            Val::I32(666),
        ).unwrap();
        let global_i64 = Global::new(
            &mut store,
            GlobalType::new(ValType::I64, Mutability::Const),
            Val::I64(666),
        ).unwrap();
        let global_f32 = Global::new(
            &mut store,
            GlobalType::new(ValType::F32, Mutability::Const),
            Val::F32(666.6),
        ).unwrap();
        let global_f64 = Global::new(
            &mut store,
            GlobalType::new(ValType::F64, Mutability::Const),
            Val::F64(666.6),
        ).unwrap();
        linker.define("spectest", "print", print);
        linker.define("spectest", "print_i32", print_i32);
        linker.define("spectest", "print_i64", print_i64);
        linker.define("spectest", "print_f32", print_f32);
        linker.define("spectest", "print_f64", print_f64);
        linker.define("spectest", "print_i32_f32", print_i32_f32);
        linker.define("spectest", "print_f64_f64", print_f64_f64);
        linker.define("spectest", "table", table);
        linker.define("spectest", "memory", memory);
        linker.define("spectest", "global_i32", global_i32);
        linker.define("spectest", "global_i64", global_i64);
        linker.define("spectest", "global_f32", global_f32);
        linker.define("spectest", "global_f64", global_f64);
        WastRunner {
            store,
            linker,
            instances_by_name: HashMap::new(),
            current_instance: None,
        }
    }

    pub fn run(&mut self, string: &str) {
        let buf = ParseBuffer::new(string).unwrap();
        let wast = parser::parse::<Wast>(&buf).unwrap();
        for directive in wast.directives {
            match directive {
                WastDirective::Wat(QuoteWat::Wat(Wat::Module(mut module))) => {
                    let name = module.id.map(|id| id.name());
                    let bytes = module.encode().unwrap();
                    self.create_instance(name, &bytes).unwrap();
                }
                WastDirective::Wat(mut wat @ QuoteWat::QuoteModule(_, _)) => {
                    let bytes = wat.encode().unwrap();
                    self.create_instance(None, &bytes).unwrap();
                }
                WastDirective::AssertMalformed {
                    module: QuoteWat::Wat(Wat::Module(mut module)),
                    ..
                } => {
                    let name = module.id.map(|id| id.name());
                    let bytes = module.encode().unwrap();
                    self.create_instance(name, &bytes).unwrap_err();
                }
                WastDirective::AssertInvalid {
                    module: QuoteWat::Wat(Wat::Module(mut module)),
                    ..
                } => {
                    let name = module.id.map(|id| id.name());
                    let bytes = module.encode().unwrap();
                    self.create_instance(name, &bytes).unwrap_err();
                }
                WastDirective::AssertUnlinkable {
                    module: Wat::Module(mut module),
                    ..
                } => {
                    let name = module.id.map(|id| id.name());
                    let bytes = module.encode().unwrap();
                    self.create_instance(name, &bytes).unwrap_err();
                }
                WastDirective::Register { name, module, .. } => {
                    let instance = self
                        .get_instance(module.map(|module| module.name()))
                        .unwrap();
                    self.register(name, instance.clone());
                }
                WastDirective::Invoke(invoke) => {
                    self.invoke(invoke).unwrap();
                }
                WastDirective::AssertTrap { exec, .. } => {
                    self.assert_trap(exec);
                }
                WastDirective::AssertReturn {
                    exec,
                    results: expected_results,
                    ..
                } => {
                    self.assert_return(exec, expected_results).unwrap();
                }
                _ => {}
            }
        }
    }

    fn get_instance(&self, name: Option<&str>) -> Option<&Instance> {
        name.map_or_else(
            || self.current_instance.as_ref(),
            |name| self.instances_by_name.get(name),
        )
    }

    fn create_instance(&mut self, name: Option<&str>, bytes: &[u8]) -> Result<(), Error> {
        let module = Arc::new(Module::new(&self.store.engine(), &bytes)?);
        let instance = self.linker.instantiate(&mut self.store, &module)?;
        if let Some(name) = name {
            self.instances_by_name
                .insert(name.to_string(), instance.clone());
        }
        self.current_instance = Some(instance);
        Ok(())
    }

    fn assert_trap(&mut self, exec: WastExecute<'_>) {
        self.execute(exec).unwrap_err();
    }

    fn assert_return(
        &mut self,
        exec: WastExecute<'_>,
        expected_results: Vec<WastRet<'_>>,
    ) -> Result<(), Error> {
        for (actual_result, expected_result) in
            self.execute(exec)?.into_iter().zip(expected_results)
        {
            assert_result(&self.store, actual_result, expected_result);
        }
        Ok(())
    }

    fn execute(&mut self, exec: WastExecute<'_>) -> Result<Vec<Val>, Error> {
        match exec {
            WastExecute::Invoke(invoke) => self.invoke(invoke),
            WastExecute::Wat(Wat::Module(mut module)) => {
                let name = module.id.map(|id| id.name());
                let bytes = module.encode().unwrap();
                self.create_instance(name, &bytes)?;
                Ok(vec![])
            }
            WastExecute::Get { module, global } => {
                Ok(vec![self.get(module.map(|module| module.name()), global)?])
            }
            _ => unimplemented!(),
        }
    }

    fn invoke(&mut self, invoke: WastInvoke<'_>) -> Result<Vec<Val>, Error> {
        let name = invoke.module.map(|module| module.name());
        let instance = self.get_instance(name).unwrap();
        let func = instance
            .exported_val(invoke.name)
            .unwrap()
            .to_func()
            .unwrap();
        let args: Vec<Val> = invoke
            .args
            .into_iter()
            .map(|arg| match arg {
                WastArg::Core(arg) => match arg {
                    WastArgCore::I32(arg) => arg.into(),
                    WastArgCore::I64(arg) => arg.into(),
                    WastArgCore::F32(arg) => f32::from_bits(arg.bits).into(),
                    WastArgCore::F64(arg) => f64::from_bits(arg.bits).into(),
                    WastArgCore::RefNull(HeapType::Func) => FuncRef::None.into(),
                    WastArgCore::RefNull(HeapType::Extern) => ExternRef::None.into(),
                    WastArgCore::RefExtern(val) => ExternRef::Some(Extern::new(&mut self.store, val)).into(),
                    _ => unimplemented!(),
                },
                _ => unimplemented!(),
            })
            .collect();
        let mut results: Vec<_> = func
            .type_(&mut self.store)
            .results()
            .iter()
            .copied()
            .map(|type_| Val::default(type_))
            .collect();
        func.call(&mut self.store, &args, &mut results)?;
        Ok(results)
    }

    fn get(&mut self, module_name: Option<&str>, global_name: &str) -> Result<Val, Error> {
        let instance = self.get_instance(module_name).unwrap();
        let global = instance.exported_global(global_name).unwrap();
        Ok(global.get(&mut self.store))
    }

    fn register(&mut self, name: &str, instance: Instance) {
        for (export_name, export_val) in instance.exports() {
            self.linker.define(name, export_name, export_val);
        }
        self.current_instance = Some(instance);
    }
}

fn assert_result(store: &Store, actual: Val, expected: WastRet<'_>) {
    match expected {
        WastRet::Core(expected) => match expected {
            WastRetCore::I32(expected) => {
                assert_eq!(actual.to_i32().unwrap(), expected)
            }
            WastRetCore::I64(expected) => {
                assert_eq!(actual.to_i64().unwrap(), expected)
            }
            WastRetCore::F32(expected) => match expected {
                NanPattern::CanonicalNan => {
                    assert!(
                        actual.to_f32().unwrap().to_bits() & 0b0_11111111_11111111111111111111111
                            == 0b0_11111111_10000000000000000000000
                    );
                }
                NanPattern::ArithmeticNan => {
                    assert!(
                        actual.to_f32().unwrap().to_bits() & 0b0_11111111_11111111111111111111111
                            >= 0b0_11111111_10000000000000000000000
                    );
                }
                NanPattern::Value(expected_result) => {
                    assert_eq!(actual.to_f32().unwrap().to_bits(), expected_result.bits)
                }
            },
            WastRetCore::F64(expected) => match expected {
                NanPattern::CanonicalNan => {
                    assert!(
                        actual.to_f64().unwrap().to_bits()
                            & 0b0_11111111111_1111111111111111111111111111111111111111111111111111
                            == 0b0_11111111111_1000000000000000000000000000000000000000000000000000
                    );
                }
                NanPattern::ArithmeticNan => {
                    assert!(
                        actual.to_f64().unwrap().to_bits()
                            & 0b0_11111111111_1111111111111111111111111111111111111111111111111111
                            >= 0b0_11111111111_1000000000000000000000000000000000000000000000000000
                    );
                }
                NanPattern::Value(expected_result) => {
                    assert_eq!(actual.to_f64().unwrap().to_bits(), expected_result.bits)
                }
            },
            WastRetCore::RefNull(Some(HeapType::Func)) => {
                assert_eq!(actual, Val::FuncRef(None));
            }
            WastRetCore::RefNull(Some(HeapType::Extern)) => {
                assert_eq!(actual, Val::ExternRef(ExternRef::None));
            }
            WastRetCore::RefExtern(expected) => {
                assert_eq!(
                    actual
                        .to_extern_ref()
                        .unwrap()
                        .map(|val| { *val.get(store).downcast_ref::<u32>().unwrap() }),
                    expected
                );
            }
            _ => unimplemented!(),
        },
        _ => unimplemented!(),
    }
}