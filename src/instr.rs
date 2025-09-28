use {
    crate::{
        decode::{Decode, DecodeError, Decoder},
        ref_::RefType,
        val::ValType,
    },
    std::collections::VecDeque,
};

macro_rules! for_each_instr {
    ($macro:ident) => {
        $macro! {
            // Control instructions
            Unreachable => visit_unreachable
            Nop => visit_nop
            Block { block_type: BlockType } => visit_block
            Loop { block_type: BlockType } => visit_loop
            If { block_type: BlockType } => visit_if
            Else => visit_else
            End => visit_end
            Br { label_idx: u32 } => visit_br
            BrIf { label_idx: u32 } => visit_br_if
            BrTableStart => visit_br_table_start
            BrTableLabel { label_idx: u32 } => visit_br_table_label
            BrTableEnd { default_label_idx: u32 } => visit_br_table_end
            Return => visit_return
            Call { func_idx: u32 } => visit_call
            CallIndirect { table_idx: u32, type_idx: u32 } => visit_call_indirect

            // Reference instructions
            RefNull { ref_type: RefType } => visit_ref_null
            RefIsNull => visit_ref_is_null
            RefFunc { func_idx: u32 } => visit_ref_func

            // Parametric instructions
            Drop => visit_drop
            Select => visit_select
            TypedSelectStart => visit_typed_select_start
            TypedSelectValType { val_type: ValType } => visit_typed_select_val_type
            TypedSelectEnd => visit_typed_select_end

            // Variable instructions
            LocalGet { local_idx: u32 } => visit_local_get
            LocalSet { local_idx: u32 } => visit_local_set
            LocalTee { local_idx: u32 } => visit_local_tee
            GlobalGet { global_idx: u32 } => visit_global_get
            GlobalSet { global_idx: u32 } => visit_global_set

            // Table instructions
            TableGet { table_idx: u32 } => visit_table_get
            TableSet { table_idx: u32 } => visit_table_set
            TableSize { table_idx: u32 } => visit_table_size
            TableGrow { table_idx: u32 } => visit_table_grow
            TableFill { table_idx: u32 } => visit_table_fill
            TableCopy { dst_table_idx: u32, src_table_idx: u32 } => visit_table_copy
            TableInit { elem_idx: u32, table_idx: u32 } => visit_table_init
            ElemDrop { elem_idx: u32 } => visit_elem_drop

            // Memory instructions
            I32Load { mem_arg: MemArg } => visit_i32_load
            I64Load { mem_arg: MemArg } => visit_i64_load
            F32Load { mem_arg: MemArg } => visit_f32_load
            F64Load { mem_arg: MemArg } => visit_f64_load
            I32Load8S { mem_arg: MemArg } => visit_i32_load8_s
            I32Load8U { mem_arg: MemArg } => visit_i32_load8_u
            I32Load16S { mem_arg: MemArg } => visit_i32_load16_s
            I32Load16U { mem_arg: MemArg } => visit_i32_load16_u
            I64Load8S { mem_arg: MemArg } => visit_i64_load8_s
            I64Load8U { mem_arg: MemArg } => visit_i64_load8_u
            I64Load16S { mem_arg: MemArg } => visit_i64_load16_s
            I64Load16U { mem_arg: MemArg } => visit_i64_load16_u
            I64Load32S { mem_arg: MemArg } => visit_i64_load32_s
            I64Load32U { mem_arg: MemArg } => visit_i64_load32_u
            I32Store { mem_arg: MemArg } => visit_i32_store
            I64Store { mem_arg: MemArg } => visit_i64_store
            F32Store { mem_arg: MemArg } => visit_f32_store
            F64Store { mem_arg: MemArg } => visit_f64_store
            I32Store8 { mem_arg: MemArg } => visit_i32_store8
            I32Store16 { mem_arg: MemArg } => visit_i32_store16
            I64Store8 { mem_arg: MemArg } => visit_i64_store8
            I64Store16 { mem_arg: MemArg } => visit_i64_store16
            I64Store32 { mem_arg: MemArg } => visit_i64_store32
            MemorySize => visit_memory_size
            MemoryGrow => visit_memory_grow
            MemoryFill => visit_memory_fill
            MemoryCopy => visit_memory_copy
            MemoryInit { data_idx: u32 } => visit_memory_init
            DataDrop { data_idx: u32 } => visit_data_drop

            // Numeric instructions
            I32Const { val: i32 } => visit_i32_const
            I64Const { val: i64 } => visit_i64_const
            F32Const { val: f32 } => visit_f32_const
            F64Const { val: f64 } => visit_f64_const

            I32Eqz => visit_i32_eqz
            I32Eq => visit_i32_eq
            I32Ne => visit_i32_ne
            I32LtS => visit_i32_lt_s
            I32LtU => visit_i32_lt_u
            I32GtS => visit_i32_gt_s
            I32GtU => visit_i32_gt_u
            I32LeS => visit_i32_le_s
            I32LeU => visit_i32_le_u
            I32GeS => visit_i32_ge_s
            I32GeU => visit_i32_ge_u

            I64Eqz => visit_i64_eqz
            I64Eq => visit_i64_eq
            I64Ne => visit_i64_ne
            I64LtS => visit_i64_lt_s
            I64LtU => visit_i64_lt_u
            I64GtS => visit_i64_gt_s
            I64GtU => visit_i64_gt_u
            I64LeS => visit_i64_le_s
            I64LeU => visit_i64_le_u
            I64GeS => visit_i64_ge_s
            I64GeU => visit_i64_ge_u

            F32Eq => visit_f32_eq
            F32Ne => visit_f32_ne
            F32Lt => visit_f32_lt
            F32Gt => visit_f32_gt
            F32Le => visit_f32_le
            F32Ge => visit_f32_ge

            F64Eq => visit_f64_eq
            F64Ne => visit_f64_ne
            F64Lt => visit_f64_lt
            F64Gt => visit_f64_gt
            F64Le => visit_f64_le
            F64Ge => visit_f64_ge

            I32Clz => visit_i32_clz
            I32Ctz => visit_i32_ctz
            I32Popcnt => visit_i32_popcnt
            I32Add => visit_i32_add
            I32Sub => visit_i32_sub
            I32Mul => visit_i32_mul
            I32DivS => visit_i32_div_s
            I32DivU => visit_i32_div_u
            I32RemS => visit_i32_rem_s
            I32RemU => visit_i32_rem_u
            I32And => visit_i32_and
            I32Or => visit_i32_or
            I32Xor => visit_i32_xor
            I32Shl => visit_i32_shl
            I32ShrS => visit_i32_shr_s
            I32ShrU => visit_i32_shr_u
            I32Rotl => visit_i32_rotl
            I32Rotr => visit_i32_rotr

            I64Clz => visit_i64_clz
            I64Ctz => visit_i64_ctz
            I64Popcnt => visit_i64_popcnt
            I64Add => visit_i64_add
            I64Sub => visit_i64_sub
            I64Mul => visit_i64_mul
            I64DivS => visit_i64_div_s
            I64DivU => visit_i64_div_u
            I64RemS => visit_i64_rem_s
            I64RemU => visit_i64_rem_u
            I64And => visit_i64_and
            I64Or => visit_i64_or
            I64Xor => visit_i64_xor
            I64Shl => visit_i64_shl
            I64ShrS => visit_i64_shr_s
            I64ShrU => visit_i64_shr_u
            I64Rotl => visit_i64_rotl
            I64Rotr => visit_i64_rotr

            F32Abs => visit_f32_abs
            F32Neg => visit_f32_neg
            F32Ceil => visit_f32_ceil
            F32Floor => visit_f32_floor
            F32Trunc => visit_f32_trunc
            F32Nearest => visit_f32_nearest
            F32Sqrt => visit_f32_sqrt
            F32Add => visit_f32_add
            F32Sub => visit_f32_sub
            F32Mul => visit_f32_mul
            F32Div => visit_f32_div
            F32Min => visit_f32_min
            F32Max => visit_f32_max
            F32Copysign => visit_f32_copysign

            F64Abs => visit_f64_abs
            F64Neg => visit_f64_neg
            F64Ceil => visit_f64_ceil
            F64Floor => visit_f64_floor
            F64Trunc => visit_f64_trunc
            F64Nearest => visit_f64_nearest
            F64Sqrt => visit_f64_sqrt
            F64Add => visit_f64_add
            F64Sub => visit_f64_sub
            F64Mul => visit_f64_mul
            F64Div => visit_f64_div
            F64Min => visit_f64_min
            F64Max => visit_f64_max
            F64Copysign => visit_f64_copysign

            I32WrapI64 => visit_i32_wrap_i64
            I32TruncF32S => visit_i32_trunc_f32_s
            I32TruncF32U => visit_i32_trunc_f32_u
            I32TruncF64S => visit_i32_trunc_f64_s
            I32TruncF64U => visit_i32_trunc_f64_u
            I64ExtendI32S => visit_i64_extend_i32_s
            I64ExtendI32U => visit_i64_extend_i32_u
            I64TruncF32S => visit_i64_trunc_f32_s
            I64TruncF32U => visit_i64_trunc_f32_u
            I64TruncF64S => visit_i64_trunc_f64_s
            I64TruncF64U => visit_i64_trunc_f64_u
            F32ConvertI32S => visit_f32_convert_i32_s
            F32ConvertI32U => visit_f32_convert_i32_u
            F32ConvertI64S => visit_f32_convert_i64_s
            F32ConvertI64U => visit_f32_convert_i64_u
            F32DemoteF64 => visit_f32_demote_f64
            F64ConvertI32S => visit_f64_convert_i32_s
            F64ConvertI32U => visit_f64_convert_i32_u
            F64ConvertI64S => visit_f64_convert_i64_s
            F64ConvertI64U => visit_f64_convert_i64_u
            F64PromoteF32 => visit_f64_promote_f32
            I32ReinterpretF32 => visit_i32_reinterpret_f32
            I64ReinterpretF64 => visit_i64_reinterpret_f64
            F32ReinterpretI32 => visit_f32_reinterpret_i32
            F64ReinterpretI64 => visit_f64_reinterpret_i64

            I32Extend8S => visit_i32_extend8_s
            I32Extend16S => visit_i32_extend16_s
            I64Extend8S => visit_i64_extend8_s
            I64Extend16S => visit_i64_extend16_s
            I64Extend32S => visit_i64_extend32_s

            I32TruncSatF32S => visit_i32_trunc_sat_f32_s
            I32TruncSatF32U => visit_i32_trunc_sat_f32_u
            I32TruncSatF64S => visit_i32_trunc_sat_f64_s
            I32TruncSatF64U => visit_i32_trunc_sat_f64_u
            I64TruncSatF32S => visit_i64_trunc_sat_f32_s
            I64TruncSatF32U => visit_i64_trunc_sat_f32_u
            I64TruncSatF64S => visit_i64_trunc_sat_f64_s
            I64TruncSatF64U => visit_i64_trunc_sat_f64_u
        }
    };
}

macro_rules! define_instr_enum {
    ($($instr:ident $({ $($arg:ident: $arg_ty:ty),* })? => $visit:ident)*) => {
        #[derive(Clone, Copy, Debug)]
        pub(crate) enum Instr {
            $($instr $( { $($arg: $arg_ty),* } )?),*
        }

        impl Instr {
            pub(crate) fn visit<V>(self, visitor: &mut V) -> Result<V::Ok, V::Error>
            where
                V: InstrVisitor,
            {
                match self {
                    $(Instr::$instr $( { $($arg),* } )? => visitor.$visit( $( $($arg),* )? )),*
                }
            }
        }
    }
}

for_each_instr!(define_instr_enum);

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

#[derive(Debug)]
pub(crate) struct InstrDecoder<'a> {
    decoder: Decoder<'a>,
    state: InstrDecoderState,
    frames: Vec<InstrDecoderFrame>,
}

impl<'a> InstrDecoder<'a> {
    pub(crate) fn new_with_allocs(bytes: &'a [u8], allocs: InstrDecoderAllocs) -> Self {
        let mut frames = allocs.frames;
        frames.clear();
        frames.push(InstrDecoderFrame::Block);
        Self {
            decoder: Decoder::new(bytes),
            state: InstrDecoderState::Start,
            frames,
        }
    }

    pub(crate) fn into_allocs(self) -> InstrDecoderAllocs {
        InstrDecoderAllocs {
            frames: self.frames,
        }
    }

    pub(crate) fn is_at_end(&self) -> bool {
        self.frames.is_empty()
    }

    pub(crate) fn decode<V>(
        &mut self,
        visitor: &mut V,
    ) -> Result<V::Ok, V::Error>
    where
        V: InstrVisitor,
    {
        match self.state {
            InstrDecoderState::Start => self.decode_start(visitor),
            InstrDecoderState::BrTable { label_count } => self.decode_br_table(label_count, visitor),
            InstrDecoderState::TypedSelect { val_type_count } => self.decode_typed_select(val_type_count, visitor),
        }
    }

    fn decode_start<V>(
        &mut self,
        visitor: &mut V,
    ) -> Result<V::Ok, V::Error>
    where
        V: InstrVisitor,
    {
        match self.decoder.read_byte()? {
            0x00 => visitor.visit_unreachable(),
            0x01 => visitor.visit_nop(),
            0x02 => {
                self.frames.push(InstrDecoderFrame::Block);
                visitor.visit_block(self.decoder.decode()?)
            },
            0x03 => {
                self.frames.push(InstrDecoderFrame::Loop);
                visitor.visit_loop(self.decoder.decode()?)
            },
            0x04 => {
                self.frames.push(InstrDecoderFrame::If);
                visitor.visit_if(self.decoder.decode()?)
            },
            0x05 => {
                if self.frames.pop() != Some(InstrDecoderFrame::If) {
                    return Err(DecodeError::new("unexpected else"))?;
                }
                self.frames.push(InstrDecoderFrame::Else);
                visitor.visit_else()
            },
            0x0B => {
                self.frames.pop();
                visitor.visit_end()
            },
            0x0C => visitor.visit_br(self.decoder.decode()?),
            0x0D => visitor.visit_br_if(self.decoder.decode()?),
            0x0E => {
                let label_count = self.decoder.decode::<u32>()?;
                self.state = InstrDecoderState::BrTable { label_count };
                visitor.visit_br_table_start()
            }
            0x0F => visitor.visit_return(),
            0x10 => visitor.visit_call(self.decoder.decode()?),
            0x11 => {
                let type_idx = self.decoder.decode()?;
                let table_idx = self.decoder.decode()?;
                visitor.visit_call_indirect(table_idx, type_idx)
            }
            0x1A => visitor.visit_drop(),
            0x1B => visitor.visit_select(),
            0x1C => {
                let val_type_count = self.decoder.decode::<u32>()?;
                if val_type_count != 1 {
                    return Err(DecodeError::new("invalid value type count"))?;
                }
                self.state = InstrDecoderState::TypedSelect { val_type_count };
                visitor.visit_typed_select_start()
            }
            0x20 => visitor.visit_local_get(self.decoder.decode()?),
            0x21 => visitor.visit_local_set(self.decoder.decode()?),
            0x22 => visitor.visit_local_tee(self.decoder.decode()?),
            0x23 => visitor.visit_global_get(self.decoder.decode()?),
            0x24 => visitor.visit_global_set(self.decoder.decode()?),
            0x25 => visitor.visit_table_get(self.decoder.decode()?),
            0x26 => visitor.visit_table_set(self.decoder.decode()?),
            0x28 => visitor.visit_i32_load(self.decoder.decode()?),
            0x29 => visitor.visit_i64_load(self.decoder.decode()?),
            0x2A => visitor.visit_f32_load(self.decoder.decode()?),
            0x2B => visitor.visit_f64_load(self.decoder.decode()?),
            0x2C => visitor.visit_i32_load8_s(self.decoder.decode()?),
            0x2D => visitor.visit_i32_load8_u(self.decoder.decode()?),
            0x2E => visitor.visit_i32_load16_s(self.decoder.decode()?),
            0x2F => visitor.visit_i32_load16_u(self.decoder.decode()?),
            0x30 => visitor.visit_i64_load8_s(self.decoder.decode()?),
            0x31 => visitor.visit_i64_load8_u(self.decoder.decode()?),
            0x32 => visitor.visit_i64_load16_s(self.decoder.decode()?),
            0x33 => visitor.visit_i64_load16_u(self.decoder.decode()?),
            0x34 => visitor.visit_i64_load32_s(self.decoder.decode()?),
            0x35 => visitor.visit_i64_load32_u(self.decoder.decode()?),
            0x36 => visitor.visit_i32_store(self.decoder.decode()?),
            0x37 => visitor.visit_i64_store(self.decoder.decode()?),
            0x38 => visitor.visit_f32_store(self.decoder.decode()?),
            0x39 => visitor.visit_f64_store(self.decoder.decode()?),
            0x3A => visitor.visit_i32_store8(self.decoder.decode()?),
            0x3B => visitor.visit_i32_store16(self.decoder.decode()?),
            0x3C => visitor.visit_i64_store8(self.decoder.decode()?),
            0x3D => visitor.visit_i64_store16(self.decoder.decode()?),
            0x3E => visitor.visit_i64_store32(self.decoder.decode()?),
            0x3F => {
                if self.decoder.read_byte()? != 0x00 {
                    return Err(DecodeError::new("expected zero byte"))?;
                }
                visitor.visit_memory_size()
            }
            0x40 => {
                if self.decoder.read_byte()? != 0x00 {
                    return Err(DecodeError::new("expected zero byte"))?;
                }
                visitor.visit_memory_grow()
            }
            0x41 => visitor.visit_i32_const(self.decoder.decode()?),
            0x42 => visitor.visit_i64_const(self.decoder.decode()?),
            0x43 => visitor.visit_f32_const(self.decoder.decode()?),
            0x44 => visitor.visit_f64_const(self.decoder.decode()?),
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
            0xD0 => visitor.visit_ref_null(self.decoder.decode()?),
            0xD1 => visitor.visit_ref_is_null(),
            0xD2 => visitor.visit_ref_func(self.decoder.decode()?),
            0xFC => match self.decoder.decode::<u32>()? {
                0 => visitor.visit_i32_trunc_sat_f32_s(),
                1 => visitor.visit_i32_trunc_sat_f32_u(),
                2 => visitor.visit_i32_trunc_sat_f64_s(),
                3 => visitor.visit_i32_trunc_sat_f64_u(),
                4 => visitor.visit_i64_trunc_sat_f32_s(),
                5 => visitor.visit_i64_trunc_sat_f32_u(),
                6 => visitor.visit_i64_trunc_sat_f64_s(),
                7 => visitor.visit_i64_trunc_sat_f64_u(),
                8 => {
                    let data_idx = self.decoder.decode()?;
                    if self.decoder.read_byte()? != 0x00 {
                        return Err(DecodeError::new("expected zero byte"))?;
                    }
                    visitor.visit_memory_init(data_idx)
                }
                9 => visitor.visit_data_drop(self.decoder.decode()?),
                10 => {
                    if self.decoder.read_byte()? != 0x00 {
                        return Err(DecodeError::new("expected zero byte"))?;
                    }
                    if self.decoder.read_byte()? != 0x00 {
                        return Err(DecodeError::new("expected zero byte"))?;
                    }
                    visitor.visit_memory_copy()
                }
                11 => {
                    if self.decoder.read_byte()? != 0x00 {
                        return Err(DecodeError::new("expected zero byte"))?;
                    }
                    visitor.visit_memory_fill()
                }
                12 => {
                    let elem_idx = self.decoder.decode()?;
                    let table_idx = self.decoder.decode()?;
                    visitor.visit_table_init(table_idx, elem_idx)
                }
                13 => visitor.visit_elem_drop(self.decoder.decode()?),
                14 => visitor.visit_table_copy(self.decoder.decode()?, self.decoder.decode()?),
                15 => visitor.visit_table_grow(self.decoder.decode()?),
                16 => visitor.visit_table_size(self.decoder.decode()?),
                17 => visitor.visit_table_fill(self.decoder.decode()?),
                _ => Err(DecodeError::new("illegal opcode"))?,
            },
            _ => Err(DecodeError::new("illegal opcode"))?,
        }
    }

    fn decode_br_table<V>(
        &mut self,
        label_count: u32,
        visitor: &mut V,
    ) -> Result<V::Ok, V::Error>
    where
        V: InstrVisitor,
    {
        if let Some(label_count) = label_count.checked_sub(1) {
            let label_idx = self.decoder.decode()?;
            self.state = InstrDecoderState::BrTable { label_count };
            visitor.visit_br_table_label(label_idx)
        } else {
            let default_label_idx = self.decoder.decode()?;
            self.state = InstrDecoderState::Start;
            visitor.visit_br_table_end(default_label_idx)
        }
    }

    fn decode_typed_select<V>(
        &mut self,
        val_type_count: u32,
        visitor: &mut V,
    ) -> Result<V::Ok, V::Error>
    where
        V: InstrVisitor,
    {
        if let Some(val_type_count) = val_type_count.checked_sub(1) {
            let val_type = self.decoder.decode()?;
            self.state = InstrDecoderState::TypedSelect { val_type_count };
            visitor.visit_typed_select_val_type(val_type)
        } else {
            self.state = InstrDecoderState::Start;
            visitor.visit_typed_select_end()
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum InstrDecoderState {
    Start,
    BrTable {
        label_count: u32,
    },
    TypedSelect {
        val_type_count: u32,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstrDecoderFrame {
    Block,
    Loop,
    If,
    Else,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct InstrDecoderAllocs {
    frames: Vec<InstrDecoderFrame>,
}

macro_rules! define_instr_visitor {
    ($($instr:ident $({ $($arg:ident: $arg_ty:ty),* })? => $visit:ident)*) => {
        pub(crate) trait InstrVisitor {
            type Ok;
            type Error: From<DecodeError>;

            $(fn $visit(&mut self $(, $($arg: $arg_ty),*)?) -> Result<Self::Ok, Self::Error>;)*
        }
    }
}

for_each_instr!(define_instr_visitor);

macro_rules! define_instr_factory {
    ($($instr:ident $({ $($arg:ident: $arg_ty:ty),* })? => $visit:ident)*) => {
        pub(crate) struct InstrFactory;

        impl InstrVisitor for InstrFactory {
            type Ok = Instr;
            type Error = DecodeError;

            $(fn $visit(&mut self $(, $($arg: $arg_ty),*)?) -> Result<Self::Ok, Self::Error> {
                Ok(Instr::$instr $( { $($arg),* } )?)
            })*
        }
    }
}

for_each_instr!(define_instr_factory);

#[derive(Debug)]
pub(crate) struct InstrStream<'a> {
    decoder: InstrDecoder<'a>,
    peeked: VecDeque<Instr>,
}

impl<'a> InstrStream<'a> {
    pub(crate) fn new_with_allocs(bytes: &'a [u8], allocs: InstrStreamAllocs) -> Self {
        let mut peeked = allocs.peeked;
        peeked.clear();
        Self {
            decoder: InstrDecoder::new_with_allocs(bytes, allocs.decoder_allocs),
            peeked,
        }
    }

    pub(crate) fn into_allocs(self) -> InstrStreamAllocs {
        InstrStreamAllocs {
            decoder_allocs: self.decoder.into_allocs(),
            peeked: self.peeked,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.decoder.is_at_end() && self.peeked.is_empty()
    }

    pub(crate) fn peek(&mut self, index: usize) -> Result<Instr, DecodeError> {
        while self.peeked.len() <= index {
            let instr = self.decoder.decode(&mut InstrFactory)?;
            self.peeked.push_back(instr);
        }
        Ok(self.peeked[index])
    }

    pub(crate) fn skip(&mut self, count: usize) -> Result<(), DecodeError> {
        if count <= self.peeked.len() {
            self.peeked.drain(..count);
        } else {
            let count = count - self.peeked.len();
            self.peeked.clear();
            for _ in 0..count {
                self.decoder.decode(&mut InstrFactory)?;
            }
        }
        Ok(())
    }

    pub(crate) fn next(&mut self) -> Result<Instr, DecodeError> {
        if let Some(instr) = self.peeked.pop_front() {
            Ok(instr)
        } else {
            self.decoder.decode(&mut InstrFactory)
        }
    }
}

#[derive(Clone, Default, Debug)]
pub(crate) struct InstrStreamAllocs {
    decoder_allocs: InstrDecoderAllocs,
    peeked: VecDeque<Instr>,
}