#[derive(Debug, Clone)]
pub enum Insn {
	LOAD64(u64),
	RETURN,
	ADD1,
	SUB1,
	TOCHAR,
	TOINT,
	NULLPRED,
	ZEROPRED,
	NOT,
	INTPRED,
	BOOLPRED,
	ADD,
	MULT,
	SUB,
	LESS,
	EQUAL,
}

pub fn unbox(qword: u64) -> Option<Insn> {
	if qword & 0x00000003 == 0x00000000 {
		Some(Insn::LOAD64(qword >> 0x2))
	} else if qword & 0xffff0000 == 0x00010000 {
		Some(Insn::RETURN)
	} else if qword & 0xffff0000 == 0x00020000 {
		Some(Insn::ADD1)
	} else if qword & 0xffff0000 == 0x00120000 {
		Some(Insn::SUB1)
	} else if qword & 0xffff0000 == 0x00220000 {
		Some(Insn::TOCHAR)
	} else if qword & 0xffff0000 == 0x00320000 {
		Some(Insn::TOINT)
	} else if qword & 0xffff0000 == 0x00420000 {
		Some(Insn::NULLPRED)
	} else if qword & 0xffff0000 == 0x00520000 {
		Some(Insn::ZEROPRED)
	} else if qword & 0xffff0000 == 0x00620000 {
		Some(Insn::NOT)
	} else if qword & 0xffff0000 == 0x00720000 {
		Some(Insn::INTPRED)
	} else if qword & 0xffff0000 == 0x00820000 {
		Some(Insn::BOOLPRED)
	} else if qword & 0xffff0000 == 0x00030000 {
		Some(Insn::ADD)
	} else if qword & 0xffff0000 == 0x00130000 {
		Some(Insn::MULT)
	} else if qword & 0xffff0000 == 0x00230000 {
		Some(Insn::SUB)
	} else if qword & 0xffff0000 == 0x00330000 {
		Some(Insn::LESS)
	} else if qword & 0xffff0000 == 0x00430000 {
		Some(Insn::EQUAL)
	} else {
		None
	}
}