use {
    crate::{
        aliasable_box::AliasableBox,
        decode::{Decode, DecodeError, Decoder},
        exec::{self, Imm, Reg, Stk, ThreadedInstr},
        ref_::RefType,
        val::ValType,
    },
    std::sync::Arc,
};

#[derive(Debug)]
pub(crate) enum Code {
    Uncompiled(UncompiledCode),
    Compiling,
    Compiled(CompiledCode),
}

#[derive(Clone, Debug)]
pub(crate) struct UncompiledCode {
    pub(crate) locals: Box<[ValType]>,
    pub(crate) expr: Arc<[u8]>,
}

impl Decode for UncompiledCode {
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
pub(crate) struct CompiledCode {
    pub(crate) max_stack_height: usize,
    pub(crate) local_count: usize,
    pub(crate) code: AliasableBox<[InstrSlot]>,
}

pub(crate) type InstrSlot = usize;

pub(crate) trait InstrVisitor {
    type Error;

    // Control instructions
    fn visit_nop(&mut self) -> Result<(), Self::Error>;
    fn visit_unreachable(&mut self) -> Result<(), Self::Error>;
    fn visit_block(&mut self, type_: BlockType) -> Result<(), Self::Error>;
    fn visit_loop(&mut self, type_: BlockType) -> Result<(), Self::Error>;
    fn visit_if(&mut self, type_: BlockType) -> Result<(), Self::Error>;
    fn visit_else(&mut self) -> Result<(), Self::Error>;
    fn visit_end(&mut self) -> Result<(), Self::Error>;
    fn visit_br(&mut self, label_idx: u32) -> Result<(), Self::Error>;
    fn visit_br_if(&mut self, label_idx: u32) -> Result<(), Self::Error>;
    fn visit_br_table(
        &mut self,
        label_idxs: &[u32],
        default_label_idx: u32,
    ) -> Result<(), Self::Error>;
    fn visit_return(&mut self) -> Result<(), Self::Error>;
    fn visit_call(&mut self, func_idx: u32) -> Result<(), Self::Error>;
    fn visit_call_indirect(&mut self, table_idx: u32, type_idx: u32) -> Result<(), Self::Error>;

    // Reference instructions
    fn visit_ref_null(&mut self, type_: RefType) -> Result<(), Self::Error>;
    fn visit_ref_is_null(&mut self) -> Result<(), Self::Error>;
    fn visit_ref_func(&mut self, func_idx: u32) -> Result<(), Self::Error>;

    // Parametric instructions
    fn visit_drop(&mut self) -> Result<(), Self::Error>;
    fn visit_select(&mut self, types_: Option<ValType>) -> Result<(), Self::Error>;

    // Variable instructions
    fn visit_local_get(&mut self, local_idx: u32) -> Result<(), Self::Error>;
    fn visit_local_set(&mut self, local_idx: u32) -> Result<(), Self::Error>;
    fn visit_local_tee(&mut self, local_idx: u32) -> Result<(), Self::Error>;
    fn visit_global_get(&mut self, global_idx: u32) -> Result<(), Self::Error>;
    fn visit_global_set(&mut self, global_idx: u32) -> Result<(), Self::Error>;

    // Table instructions
    fn visit_table_get(&mut self, table_idx: u32) -> Result<(), Self::Error>;
    fn visit_table_set(&mut self, table_idx: u32) -> Result<(), Self::Error>;
    fn visit_table_size(&mut self, table_idx: u32) -> Result<(), Self::Error>;
    fn visit_table_grow(&mut self, table_idx: u32) -> Result<(), Self::Error>;
    fn visit_table_fill(&mut self, table_idx: u32) -> Result<(), Self::Error>;
    fn visit_table_copy(
        &mut self,
        dst_table_idx: u32,
        src_table_idx: u32,
    ) -> Result<(), Self::Error>;
    fn visit_table_init(&mut self, table_idx: u32, elem_idx: u32) -> Result<(), Self::Error>;
    fn visit_elem_drop(&mut self, elem_idx: u32) -> Result<(), Self::Error>;

    // Memory instructions
    fn visit_load(&mut self, arg: MemArg, info: LoadInfo) -> Result<(), Self::Error>;
    fn visit_store(&mut self, arg: MemArg, info: StoreInfo) -> Result<(), Self::Error>;
    fn visit_memory_size(&mut self) -> Result<(), Self::Error>;
    fn visit_memory_grow(&mut self) -> Result<(), Self::Error>;
    fn visit_memory_fill(&mut self) -> Result<(), Self::Error>;
    fn visit_memory_copy(&mut self) -> Result<(), Self::Error>;
    fn visit_memory_init(&mut self, data_idx: u32) -> Result<(), Self::Error>;
    fn visit_data_drop(&mut self, data_idx: u32) -> Result<(), Self::Error>;

    // Numeric instructions
    fn visit_i32_const(&mut self, val: i32) -> Result<(), Self::Error>;
    fn visit_i64_const(&mut self, val: i64) -> Result<(), Self::Error>;
    fn visit_f32_const(&mut self, val: f32) -> Result<(), Self::Error>;
    fn visit_f64_const(&mut self, val: f64) -> Result<(), Self::Error>;

    fn visit_i32_eqz(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_eq(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_ne(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_lt_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_lt_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_gt_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_gt_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_le_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_le_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_ge_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_ge_u(&mut self) -> Result<(), Self::Error>;

    fn visit_i64_eqz(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_eq(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_ne(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_lt_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_lt_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_gt_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_gt_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_le_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_le_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_ge_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_ge_u(&mut self) -> Result<(), Self::Error>;

    fn visit_f32_eq(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_ne(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_lt(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_gt(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_le(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_ge(&mut self) -> Result<(), Self::Error>;
    
    fn visit_f64_eq(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_ne(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_lt(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_gt(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_le(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_ge(&mut self) -> Result<(), Self::Error>;

    fn visit_i32_clz(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_ctz(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_popcnt(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_add(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_sub(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_mul(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_div_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_div_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_rem_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_rem_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_and(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_or(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_xor(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_shl(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_shr_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_shr_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_rotl(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_rotr(&mut self) -> Result<(), Self::Error>;

    fn visit_i64_clz(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_ctz(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_popcnt(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_add(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_sub(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_mul(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_div_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_div_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_rem_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_rem_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_and(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_or(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_xor(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_shl(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_shr_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_shr_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_rotl(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_rotr(&mut self) -> Result<(), Self::Error>;

    fn visit_f32_abs(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_neg(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_ceil(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_floor(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_trunc(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_nearest(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_sqrt(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_add(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_sub(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_mul(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_div(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_min(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_max(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_copysign(&mut self) -> Result<(), Self::Error>;

    fn visit_f64_abs(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_neg(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_ceil(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_floor(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_trunc(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_nearest(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_sqrt(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_add(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_sub(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_mul(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_div(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_min(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_max(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_copysign(&mut self) -> Result<(), Self::Error>;

    fn visit_i32_wrap_i64(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_trunc_f32_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_trunc_f32_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_trunc_f64_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_trunc_f64_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_extend_i32_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_extend_i32_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_f32_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_f32_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_f64_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_f64_u(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_convert_i32_s(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_convert_i32_u(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_convert_i64_s(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_convert_i64_u(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_demote_f64(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_convert_i32_s(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_convert_i32_u(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_convert_i64_s(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_convert_i64_u(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_promote_f32(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_reinterpret_f32(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_reinterpret_f64(&mut self) -> Result<(), Self::Error>;
    fn visit_f32_reinterpret_i32(&mut self) -> Result<(), Self::Error>;
    fn visit_f64_reinterpret_i64(&mut self) -> Result<(), Self::Error>;

    fn visit_i32_extend8_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_extend16_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_extend8_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_extend16_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_extend32_s(&mut self) -> Result<(), Self::Error>;

    fn visit_i32_trunc_sat_f32_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_trunc_sat_f32_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_trunc_sat_f64_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i32_trunc_sat_f64_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_sat_f32_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_sat_f32_u(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_sat_f64_s(&mut self) -> Result<(), Self::Error>;
    fn visit_i64_trunc_sat_f64_u(&mut self) -> Result<(), Self::Error>;

    fn visit_un_op(&mut self, info: UnOpInfo) -> Result<(), Self::Error>;
    fn visit_bin_op(&mut self, info: BinOpInfo) -> Result<(), Self::Error>;
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BlockType {
    TypeIdx(u32),
    ValType(Option<ValType>),
}

impl Decode for BlockType {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        fn decode_i33_tail(decoder: &mut Decoder<'_>, mut value: i64) -> Result<i64, DecodeError> {
            let mut shift = 0;
            loop {
                let byte = decoder.read_byte()?;
                if shift >= 26 && byte >> 33 - shift != 0 {
                    let sign = (byte << 1) as i8 >> (33 - shift);
                    if byte & 0x80 != 0x00 || sign != 0 && sign != -1 {
                        return Err(DecodeError::new("malformed s33"));
                    }
                }
                value |= ((byte & 0x7F) as i64) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            let shift = 58 - shift.min(26);
            Ok(value << shift >> shift)
        }

        match decoder.read_byte()? {
            0x40 => Ok(BlockType::ValType(None)),
            0x7F => Ok(BlockType::ValType(Some(ValType::I32))),
            0x7E => Ok(BlockType::ValType(Some(ValType::I64))),
            0x7D => Ok(BlockType::ValType(Some(ValType::F32))),
            0x7C => Ok(BlockType::ValType(Some(ValType::F64))),
            0x70 => Ok(BlockType::ValType(Some(ValType::FuncRef))),
            0x6F => Ok(BlockType::ValType(Some(ValType::ExternRef))),
            byte => {
                let value = (byte & 0x7F) as i64;
                let value = if byte & 0x80 == 0x00 {
                    value
                } else {
                    decode_i33_tail(decoder, value)?
                };
                if value < 0 {
                    return Err(DecodeError::new(""));
                }
                Ok(BlockType::TypeIdx(value as u32))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MemArg {
    pub(crate) align: u32,
    pub(crate) offset: u32,
}

impl Decode for MemArg {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            align: decoder.decode()?,
            offset: decoder.decode()?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LoadInfo {
    pub(crate) max_align: u32,
    pub(crate) op: UnOpInfo,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StoreInfo {
    pub(crate) max_align: u32,
    pub(crate) op: BinOpInfo,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct UnOpInfo {
    pub(crate) _name: &'static str,
    pub(crate) input_type: ValType,
    pub(crate) output_type: Option<ValType>,
    pub(crate) instr_s: ThreadedInstr,
    pub(crate) instr_r: ThreadedInstr,
    pub(crate) instr_i: Option<ThreadedInstr>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BinOpInfo {
    pub(crate) _name: &'static str,
    pub(crate) input_type_0: ValType,
    pub(crate) input_type_1: ValType,
    pub(crate) output_type: Option<ValType>,
    pub(crate) instr_ss: ThreadedInstr,
    pub(crate) instr_rs: ThreadedInstr,
    pub(crate) instr_is: ThreadedInstr,
    pub(crate) instr_ir: ThreadedInstr,
    pub(crate) instr_ii: Option<ThreadedInstr>,
    pub(crate) instr_sr: ThreadedInstr,
    pub(crate) instr_si: ThreadedInstr,
    pub(crate) instr_ri: ThreadedInstr,
    pub(crate) instr_rr: Option<ThreadedInstr>,
}

pub(crate) fn decode_instr<V>(
    decoder: &mut Decoder<'_>,
    label_idxs: &mut Vec<u32>,
    visitor: &mut V,
) -> Result<(), V::Error>
where
    V: InstrVisitor,
    V::Error: From<DecodeError>,
{
    match decoder.read_byte()? {
        0x00 => visitor.visit_unreachable(),
        0x01 => visitor.visit_nop(),
        0x02 => visitor.visit_block(decoder.decode()?),
        0x03 => visitor.visit_loop(decoder.decode()?),
        0x04 => visitor.visit_if(decoder.decode()?),
        0x05 => visitor.visit_else(),
        0x0B => visitor.visit_end(),
        0x0C => visitor.visit_br(decoder.decode()?),
        0x0D => visitor.visit_br_if(decoder.decode()?),
        0x0E => {
            label_idxs.clear();
            for label_idx in decoder.decode_iter()? {
                label_idxs.push(label_idx?);
            }
            visitor.visit_br_table(&label_idxs, decoder.decode()?)?;
            Ok(())
        }
        0x0F => visitor.visit_return(),
        0x10 => visitor.visit_call(decoder.decode()?),
        0x11 => {
            let type_idx = decoder.decode()?;
            let table_idx = decoder.decode()?;
            visitor.visit_call_indirect(table_idx, type_idx)
        }
        0x1A => visitor.visit_drop(),
        0x1B => visitor.visit_select(None),
        0x1C => {
            if decoder.decode::<u32>()? != 1 {
                return Err(DecodeError::new(""))?;
            }
            visitor.visit_select(Some(decoder.decode()?))?;
            Ok(())
        }
        0x20 => visitor.visit_local_get(decoder.decode()?),
        0x21 => visitor.visit_local_set(decoder.decode()?),
        0x22 => visitor.visit_local_tee(decoder.decode()?),
        0x23 => visitor.visit_global_get(decoder.decode()?),
        0x24 => visitor.visit_global_set(decoder.decode()?),
        0x25 => visitor.visit_table_get(decoder.decode()?),
        0x26 => visitor.visit_table_set(decoder.decode()?),
        0x28 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 2,
                op: UnOpInfo {
                    _name: "i32_load",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I32),
                    instr_s: exec::load::<i32, Stk, Reg>,
                    instr_r: exec::load::<i32, Reg, Reg>,
                    instr_i: Some(exec::load::<i32, Imm, Reg>),
                },
            },
        ),
        0x29 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 3,
                op: UnOpInfo {
                    _name: "i64_load",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I64),
                    instr_s: exec::load::<i64, Stk, Reg>,
                    instr_r: exec::load::<i64, Reg, Reg>,
                    instr_i: Some(exec::load::<i64, Imm, Reg>),
                },
            },
        ),
        0x2A => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 2,
                op: UnOpInfo {
                    _name: "f32_load",
                    input_type: ValType::I32,
                    output_type: Some(ValType::F32),
                    instr_s: exec::load::<f32, Stk, Reg>,
                    instr_r: exec::load::<f32, Reg, Reg>,
                    instr_i: Some(exec::load::<f32, Imm, Reg>),
                },
            },
        ),
        0x2B => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 3,
                op: UnOpInfo {
                    _name: "f64_load",
                    input_type: ValType::I32,
                    output_type: Some(ValType::F64),
                    instr_s: exec::load::<f64, Stk, Reg>,
                    instr_r: exec::load::<f64, Reg, Reg>,
                    instr_i: Some(exec::load::<f64, Imm, Reg>),
                },
            },
        ),
        0x2C => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 0,
                op: UnOpInfo {
                    _name: "i32_load8_s",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I32),
                    instr_s: exec::load_n::<i32, i8, Stk, Reg>,
                    instr_r: exec::load_n::<i32, i8, Reg, Reg>,
                    instr_i: Some(exec::load_n::<i32, i8, Imm, Reg>),
                },
            },
        ),
        0x2D => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 0,
                op: UnOpInfo {
                    _name: "i32_load8_u",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I32),
                    instr_s: exec::load_n::<u32, u8, Stk, Reg>,
                    instr_r: exec::load_n::<u32, u8, Reg, Reg>,
                    instr_i: Some(exec::load_n::<u32, u8, Imm, Reg>),
                },
            },
        ),
        0x2E => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 1,
                op: UnOpInfo {
                    _name: "i32_load16_s",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I32),
                    instr_s: exec::load_n::<i32, i16, Stk, Reg>,
                    instr_r: exec::load_n::<i32, i16, Reg, Reg>,
                    instr_i: Some(exec::load_n::<i32, i16, Imm, Reg>),
                },
            },
        ),
        0x2F => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 1,
                op: UnOpInfo {
                    _name: "i32_load16_u",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I32),
                    instr_s: exec::load_n::<u32, u16, Stk, Reg>,
                    instr_r: exec::load_n::<u32, u16, Reg, Reg>,
                    instr_i: Some(exec::load_n::<u32, u16, Imm, Reg>),
                },
            },
        ),
        0x30 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 0,
                op: UnOpInfo {
                    _name: "i64_load8_s",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I64),
                    instr_s: exec::load_n::<i64, i8, Stk, Reg>,
                    instr_r: exec::load_n::<i64, i8, Reg, Reg>,
                    instr_i: Some(exec::load_n::<i64, i8, Imm, Reg>),
                },
            },
        ),
        0x31 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 0,
                op: UnOpInfo {
                    _name: "i64_load8_u",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I64),
                    instr_s: exec::load_n::<u64, u8, Stk, Reg>,
                    instr_r: exec::load_n::<u64, u8, Reg, Reg>,
                    instr_i: Some(exec::load_n::<u64, u8, Imm, Reg>),
                },
            },
        ),
        0x32 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 1,
                op: UnOpInfo {
                    _name: "i64_load16_s",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I64),
                    instr_s: exec::load_n::<i64, i16, Stk, Reg>,
                    instr_r: exec::load_n::<i64, i16, Reg, Reg>,
                    instr_i: Some(exec::load_n::<i64, i16, Imm, Reg>),
                },
            },
        ),
        0x33 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 1,
                op: UnOpInfo {
                    _name: "i64_load16_u",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I64),
                    instr_s: exec::load_n::<u64, u16, Stk, Reg>,
                    instr_r: exec::load_n::<u64, u16, Reg, Reg>,
                    instr_i: Some(exec::load_n::<u64, u16, Imm, Reg>),
                },
            },
        ),
        0x34 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 2,
                op: UnOpInfo {
                    _name: "i64_load32_s",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I64),
                    instr_s: exec::load_n::<i64, i32, Stk, Reg>,
                    instr_r: exec::load_n::<i64, i32, Reg, Reg>,
                    instr_i: Some(exec::load_n::<i64, i32, Imm, Reg>),
                },
            },
        ),
        0x35 => visitor.visit_load(
            decoder.decode()?,
            LoadInfo {
                max_align: 2,
                op: UnOpInfo {
                    _name: "i64_load32_u",
                    input_type: ValType::I32,
                    output_type: Some(ValType::I64),
                    instr_s: exec::load_n::<u64, u32, Stk, Reg>,
                    instr_r: exec::load_n::<u64, u32, Reg, Reg>,
                    instr_i: Some(exec::load_n::<u64, u32, Imm, Reg>),
                },
            },
        ),
        0x36 => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 2,
                op: BinOpInfo {
                    _name: "i32_store",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::I32,
                    output_type: None,
                    instr_ss: exec::store::<i32, Stk, Stk>,
                    instr_rs: exec::store::<i32, Reg, Stk>,
                    instr_is: exec::store::<i32, Imm, Stk>,
                    instr_ir: exec::store::<i32, Imm, Reg>,
                    instr_ii: Some(exec::store::<i32, Imm, Imm>),
                    instr_sr: exec::store::<i32, Stk, Reg>,
                    instr_si: exec::store::<i32, Stk, Imm>,
                    instr_ri: exec::store::<i32, Reg, Imm>,
                    instr_rr: None,
                },
            },
        ),
        0x37 => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 3,
                op: BinOpInfo {
                    _name: "i64_store",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::I64,
                    output_type: None,
                    instr_ss: exec::store::<i64, Stk, Stk>,
                    instr_rs: exec::store::<i64, Reg, Stk>,
                    instr_is: exec::store::<i64, Imm, Stk>,
                    instr_ir: exec::store::<i64, Imm, Reg>,
                    instr_ii: Some(exec::store::<i64, Imm, Imm>),
                    instr_sr: exec::store::<i64, Stk, Reg>,
                    instr_si: exec::store::<i64, Stk, Imm>,
                    instr_ri: exec::store::<i64, Reg, Imm>,
                    instr_rr: None,
                },
            },
        ),
        0x38 => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 2,
                op: BinOpInfo {
                    _name: "f32_store",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::F32,
                    output_type: None,
                    instr_ss: exec::store::<f32, Stk, Stk>,
                    instr_rs: exec::store::<f32, Reg, Stk>,
                    instr_is: exec::store::<f32, Imm, Stk>,
                    instr_ir: exec::store::<f32, Imm, Reg>,
                    instr_ii: Some(exec::store::<f32, Imm, Imm>),
                    instr_sr: exec::store::<f32, Stk, Reg>,
                    instr_si: exec::store::<f32, Stk, Imm>,
                    instr_ri: exec::store::<f32, Reg, Imm>,
                    instr_rr: Some(exec::store::<f32, Reg, Reg>),
                },
            },
        ),
        0x39 => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 3,
                op: BinOpInfo {
                    _name: "f64_store",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::F64,
                    output_type: None,
                    instr_ss: exec::store::<f64, Stk, Stk>,
                    instr_rs: exec::store::<f64, Reg, Stk>,
                    instr_is: exec::store::<f64, Imm, Stk>,
                    instr_ir: exec::store::<f64, Imm, Reg>,
                    instr_ii: Some(exec::store::<f64, Imm, Imm>),
                    instr_sr: exec::store::<f64, Stk, Reg>,
                    instr_si: exec::store::<f64, Stk, Imm>,
                    instr_ri: exec::store::<f64, Reg, Imm>,
                    instr_rr: Some(exec::store::<f64, Reg, Reg>),
                },
            },
        ),
        0x3A => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 0,
                op: BinOpInfo {
                    _name: "i32_store8",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::I32,
                    output_type: None,
                    instr_ss: exec::store_n::<i32, i8, Stk, Stk>,
                    instr_rs: exec::store_n::<i32, i8, Reg, Stk>,
                    instr_is: exec::store_n::<i32, i8, Imm, Stk>,
                    instr_ir: exec::store_n::<i32, i8, Imm, Reg>,
                    instr_ii: Some(exec::store_n::<i32, i8, Imm, Imm>),
                    instr_sr: exec::store_n::<i32, i8, Stk, Reg>,
                    instr_si: exec::store_n::<i32, i8, Stk, Imm>,
                    instr_ri: exec::store_n::<i32, i8, Reg, Imm>,
                    instr_rr: None,
                },
            },
        ),
        0x3B => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 1,
                op: BinOpInfo {
                    _name: "i32_store16",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::I32,
                    output_type: None,
                    instr_ss: exec::store_n::<i32, i16, Stk, Stk>,
                    instr_rs: exec::store_n::<i32, i16, Reg, Stk>,
                    instr_is: exec::store_n::<i32, i16, Imm, Stk>,
                    instr_ir: exec::store_n::<i32, i16, Imm, Reg>,
                    instr_ii: Some(exec::store_n::<i32, i16, Imm, Imm>),
                    instr_sr: exec::store_n::<i32, i16, Stk, Reg>,
                    instr_si: exec::store_n::<i32, i16, Stk, Imm>,
                    instr_ri: exec::store_n::<i32, i16, Reg, Imm>,
                    instr_rr: None,
                },
            },
        ),
        0x3C => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 0,
                op: BinOpInfo {
                    _name: "i64_store8",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::I64,
                    output_type: None,
                    instr_ss: exec::store_n::<i64, i8, Stk, Stk>,
                    instr_rs: exec::store_n::<i64, i8, Reg, Stk>,
                    instr_is: exec::store_n::<i64, i8, Imm, Stk>,
                    instr_ir: exec::store_n::<i64, i8, Imm, Reg>,
                    instr_ii: Some(exec::store_n::<i64, i8, Imm, Imm>),
                    instr_sr: exec::store_n::<i64, i8, Stk, Reg>,
                    instr_si: exec::store_n::<i64, i8, Stk, Imm>,
                    instr_ri: exec::store_n::<i64, i8, Reg, Imm>,
                    instr_rr: None,
                },
            },
        ),
        0x3D => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 1,
                op: BinOpInfo {
                    _name: "i64_store16",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::I64,
                    output_type: None,
                    instr_ss: exec::store_n::<i64, i16, Stk, Stk>,
                    instr_rs: exec::store_n::<i64, i16, Reg, Stk>,
                    instr_is: exec::store_n::<i64, i16, Imm, Stk>,
                    instr_ir: exec::store_n::<i64, i16, Imm, Reg>,
                    instr_ii: Some(exec::store_n::<i64, i16, Imm, Imm>),
                    instr_sr: exec::store_n::<i64, i16, Stk, Reg>,
                    instr_si: exec::store_n::<i64, i16, Stk, Imm>,
                    instr_ri: exec::store_n::<i64, i16, Reg, Imm>,
                    instr_rr: None,
                },
            },
        ),
        0x3E => visitor.visit_store(
            decoder.decode()?,
            StoreInfo {
                max_align: 2,
                op: BinOpInfo {
                    _name: "i64_store32",
                    input_type_0: ValType::I32,
                    input_type_1: ValType::I64,
                    output_type: None,
                    instr_ss: exec::store_n::<i64, i32, Stk, Stk>,
                    instr_rs: exec::store_n::<i64, i32, Reg, Stk>,
                    instr_is: exec::store_n::<i64, i32, Imm, Stk>,
                    instr_ir: exec::store_n::<i64, i32, Imm, Reg>,
                    instr_ii: Some(exec::store_n::<i64, i32, Imm, Imm>),
                    instr_sr: exec::store_n::<i64, i32, Stk, Reg>,
                    instr_si: exec::store_n::<i64, i32, Stk, Imm>,
                    instr_ri: exec::store_n::<i64, i32, Reg, Imm>,
                    instr_rr: None,
                },
            },
        ),
        0x3F => {
            if decoder.read_byte()? != 0x00 {
                return Err(DecodeError::new("expected zero byte"))?;
            }
            visitor.visit_memory_size()
        }
        0x40 => {
            if decoder.read_byte()? != 0x00 {
                return Err(DecodeError::new("expected zero byte"))?;
            }
            visitor.visit_memory_grow()
        }
        0x41 => visitor.visit_i32_const(decoder.decode()?),
        0x42 => visitor.visit_i64_const(decoder.decode()?),
        0x43 => visitor.visit_f32_const(decoder.decode()?),
        0x44 => visitor.visit_f64_const(decoder.decode()?),
        0x45 => visitor.visit_i32_eqz(),
        0x46 => visitor.visit_i32_eq(),
        0x47 => visitor.visit_i32_ne(),
        0x48 => visitor.visit_i32_lt_s(),
        0x49 => visitor.visit_i32_lt_u(),
        0x4A => visitor.visit_i32_gt_s(),
        0x4B => visitor.visit_i32_gt_u(),
        0x4C => visitor.visit_i32_le_s(),
        0x4D => visitor.visit_i32_le_u(),
        0x4E => visitor.visit_i32_ge_s(),
        0x4F => visitor.visit_i32_ge_u(),
        0x50 => visitor.visit_i64_eqz(),
        0x51 => visitor.visit_i64_eq(),
        0x52 => visitor.visit_i64_ne(),
        0x53 => visitor.visit_i64_lt_s(),
        0x54 => visitor.visit_i64_lt_u(),
        0x55 => visitor.visit_i64_gt_s(),
        0x56 => visitor.visit_i64_gt_u(),
        0x57 => visitor.visit_i64_le_s(),
        0x58 => visitor.visit_i64_le_u(),
        0x59 => visitor.visit_i64_ge_s(),
        0x5A => visitor.visit_i64_ge_u(),
        0x5B => visitor.visit_f32_eq(),
        0x5C => visitor.visit_f32_ne(),
        0x5D => visitor.visit_f32_lt(),
        0x5E => visitor.visit_f32_gt(),
        0x5F => visitor.visit_f32_le(),
        0x60 => visitor.visit_f32_ge(),
        0x61 => visitor.visit_f64_eq(),
        0x62 => visitor.visit_f64_ne(),
        0x63 => visitor.visit_f64_lt(),
        0x64 => visitor.visit_f64_gt(),
        0x65 => visitor.visit_f64_le(),
        0x66 => visitor.visit_f64_ge(),
        0x67 => visitor.visit_i32_clz(),
        0x68 => visitor.visit_i32_ctz(),
        0x69 => visitor.visit_i32_popcnt(),
        0x6A => visitor.visit_i32_add(),
        0x6B => visitor.visit_i32_sub(),
        0x6C => visitor.visit_i32_mul(),
        0x6D => visitor.visit_i32_div_s(),
        0x6E => visitor.visit_i32_div_u(),
        0x6F => visitor.visit_i32_rem_s(),
        0x70 => visitor.visit_i32_rem_u(),
        0x71 => visitor.visit_i32_and(),
        0x72 => visitor.visit_i32_or(),
        0x73 => visitor.visit_i32_xor(),
        0x74 => visitor.visit_i32_shl(),
        0x75 => visitor.visit_i32_shr_s(),
        0x76 => visitor.visit_i32_shr_u(),
        0x77 => visitor.visit_i32_rotl(),
        0x78 => visitor.visit_i32_rotr(),
        0x79 => visitor.visit_i64_clz(),
        0x7A => visitor.visit_i64_ctz(),
        0x7B => visitor.visit_i64_popcnt(),
        0x7C => visitor.visit_i64_add(),
        0x7D => visitor.visit_i64_sub(),
        0x7E => visitor.visit_i64_mul(),
        0x7F => visitor.visit_i64_div_s(),
        0x80 => visitor.visit_i64_div_u(),
        0x81 => visitor.visit_i64_rem_s(),
        0x82 => visitor.visit_i64_rem_u(),
        0x83 => visitor.visit_i64_and(),
        0x84 => visitor.visit_i64_or(),
        0x85 => visitor.visit_i64_xor(),
        0x86 => visitor.visit_i64_shl(),
        0x87 => visitor.visit_i64_shr_s(),
        0x88 => visitor.visit_i64_shr_u(),
        0x89 => visitor.visit_i64_rotl(),
        0x8A => visitor.visit_i64_rotr(),
        0x8B => visitor.visit_f32_abs(),
        0x8C => visitor.visit_f32_neg(),
        0x8D => visitor.visit_f32_ceil(),
        0x8E => visitor.visit_f32_floor(),
        0x8F => visitor.visit_f32_trunc(),
        0x90 => visitor.visit_f32_nearest(),
        0x91 => visitor.visit_f32_sqrt(),
        0x92 => visitor.visit_f32_add(),
        0x93 => visitor.visit_f32_sub(),
        0x94 => visitor.visit_f32_mul(),
        0x95 => visitor.visit_f32_div(),
        0x96 => visitor.visit_f32_min(),
        0x97 => visitor.visit_f32_max(),
        0x98 => visitor.visit_f32_copysign(),
        0x99 => visitor.visit_f64_abs(),
        0x9A => visitor.visit_f64_neg(),
        0x9B => visitor.visit_f64_ceil(),
        0x9C => visitor.visit_f64_floor(),
        0x9D => visitor.visit_f64_trunc(),
        0x9E => visitor.visit_f64_nearest(),
        0x9F => visitor.visit_f64_sqrt(),
        0xA0 => visitor.visit_f64_add(),
        0xA1 => visitor.visit_f64_sub(),
        0xA2 => visitor.visit_f64_mul(),
        0xA3 => visitor.visit_f64_div(),
        0xA4 => visitor.visit_f64_min(),
        0xA5 => visitor.visit_f64_max(),
        0xA6 => visitor.visit_f64_copysign(),
        0xA7 => visitor.visit_i32_wrap_i64(),
        0xA8 => visitor.visit_i32_trunc_f32_s(),
        0xA9 => visitor.visit_i32_trunc_f32_u(),
        0xAA => visitor.visit_i32_trunc_f64_s(),
        0xAB => visitor.visit_i32_trunc_f64_u(),
        0xAC => visitor.visit_i64_extend_i32_s(),
        0xAD => visitor.visit_i64_extend_i32_u(),
        0xAE => visitor.visit_i64_trunc_f32_s(),
        0xAF => visitor.visit_i64_trunc_f32_u(),
        0xB0 => visitor.visit_i64_trunc_f64_s(),
        0xB1 => visitor.visit_i64_trunc_f64_u(),
        0xB2 => visitor.visit_f32_convert_i32_s(),
        0xB3 => visitor.visit_f32_convert_i32_u(),
        0xB4 => visitor.visit_f32_convert_i64_s(),
        0xB5 => visitor.visit_f32_convert_i64_u(),
        0xB6 => visitor.visit_f32_demote_f64(),
        0xB7 => visitor.visit_f64_convert_i32_s(),
        0xB8 => visitor.visit_f64_convert_i32_u(),
        0xB9 => visitor.visit_f64_convert_i64_s(),
        0xBA => visitor.visit_f64_convert_i64_u(),
        0xBB => visitor.visit_f64_promote_f32(),
        0xBC => visitor.visit_i32_reinterpret_f32(),
        0xBD => visitor.visit_i64_reinterpret_f64(),
        0xBE => visitor.visit_f32_reinterpret_i32(),
        0xBF => visitor.visit_f64_reinterpret_i64(),
        0xC0 => visitor.visit_i32_extend8_s(),
        0xC1 => visitor.visit_i32_extend16_s(),
        0xC2 => visitor.visit_i64_extend8_s(),
        0xC3 => visitor.visit_i64_extend16_s(),
        0xC4 => visitor.visit_i64_extend32_s(),
        0xD0 => visitor.visit_ref_null(decoder.decode()?),
        0xD1 => visitor.visit_ref_is_null(),
        0xD2 => visitor.visit_ref_func(decoder.decode()?),
        0xFC => match decoder.decode::<u32>()? {
            0 => visitor.visit_i32_trunc_sat_f32_s(),
            1 => visitor.visit_i32_trunc_sat_f32_u(),
            2 => visitor.visit_i32_trunc_sat_f64_s(),
            3 => visitor.visit_i32_trunc_sat_f64_u(),
            4 => visitor.visit_i64_trunc_sat_f32_s(),
            5 => visitor.visit_i64_trunc_sat_f32_u(),
            6 => visitor.visit_i64_trunc_sat_f64_s(),
            7 => visitor.visit_i64_trunc_sat_f64_u(),
            8 => {
                let data_idx = decoder.decode()?;
                if decoder.read_byte()? != 0x00 {
                    return Err(DecodeError::new("expected zero byte"))?;
                }
                visitor.visit_memory_init(data_idx)
            }
            9 => visitor.visit_data_drop(decoder.decode()?),
            10 => {
                if decoder.read_byte()? != 0x00 {
                    return Err(DecodeError::new("expected zero byte"))?;
                }
                if decoder.read_byte()? != 0x00 {
                    return Err(DecodeError::new("expected zero byte"))?;
                }
                visitor.visit_memory_copy()
            }
            11 => {
                if decoder.read_byte()? != 0x00 {
                    return Err(DecodeError::new("expected zero byte"))?;
                }
                visitor.visit_memory_fill()
            }
            12 => {
                let elem_idx = decoder.decode()?;
                let table_idx = decoder.decode()?;
                visitor.visit_table_init(table_idx, elem_idx)
            }
            13 => visitor.visit_elem_drop(decoder.decode()?),
            14 => visitor.visit_table_copy(decoder.decode()?, decoder.decode()?),
            15 => visitor.visit_table_grow(decoder.decode()?),
            16 => visitor.visit_table_size(decoder.decode()?),
            17 => visitor.visit_table_fill(decoder.decode()?),
            _ => Err(DecodeError::new("illegal opcode"))?,
        },
        _ => Err(DecodeError::new("illegal opcode"))?,
    }
}
