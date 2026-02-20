// TODO: Load instruction from an external file, this should be generated python
// TODO: Same thing for unboxing (and aligning this to Insns)
#[derive(Debug, Clone)]
pub enum Insn {
    LOAD64(u64),
    ADD,
    SUB,
    EQUAL,
    // TODO: Replace binop and unop with enum types rather than string
    // These enum types correspond to the python generated stuff
    RETURN,
}

// TODO: Fill this in and maybe fix the type
// TODO: Move unbox to the generated python function
pub fn unbox(qword: u64) -> Option<Insn> {
    if qword & 0xff == 0x11 {
        qword >> 2
    }
    match qword {
        qword if qword & 0xff == 0x11 => Some(Insn::ADD),
        _ => None,
    }
}

// TODO: Add builder for Insns from python
// TODO: Include build script for python in justfile whenever anything happens
// TODO: BONUS IDEA! Have the builder happen anytime compiler/main.py is updated
// TODO: Add box test about values that are too big for LOAD64 (use more than 62 bits)
// TODO: Similar tests for function arguments
