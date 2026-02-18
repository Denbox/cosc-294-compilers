use std::fmt;

// TODO: Load instruction from an external file, this should be generated python
// TODO: Same thing for unboxing (and aligning this to Insns)
#[derive(Debug, Clone)]
enum Insn {
    LOAD64(u64),
    // TODO: Replace binop and unop with enum types rather than string
    // These enum types correspond to the python generated stuff
    BINOP(String),
    UNOP(String),
    RETURN,
}

// struct Pointer {

// }

// TODO: Fill this in and maybe fix the type
// TODO: Move unbox to the generated python function
fn unbox(qword: u64) -> Option<Insn> {
    // TODO: Replace me to actually be a valid unbox
    Some(Insn::RETURN)
}

#[derive(Debug)]
enum MachineError {
    EmptyStackPop,
    InvalidAddress,
    InvalidOpcode,
    InvalidOperation,
}

impl fmt::Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid first item to double")
    }
}

struct Machine {
    pc: u64, // 64 bits is far more than necessary but it's okay
    code: Vec<u64>,
    stack: Vec<Insn>,
}

impl Machine {
    pub fn new(code: Vec<u64>) -> Self {
        Self {
            pc: 0,
            code,
            stack: vec![],
        }
    }

    fn push(&mut self, insn: Insn) {
        self.stack.push(insn);
    }

    fn pop(&mut self) -> Result<Insn, MachineError> {
        match self.stack.pop() {
            Some(insn) => Ok(insn),
            None => Err(MachineError::EmptyStackPop),
        }
    }

    fn read_insn(&mut self, index: u64) -> Result<Insn, MachineError> {
        match self.code.get(index as usize) {
            Some(bits) => unbox(*bits).ok_or(MachineError::InvalidOpcode),
            None => Err(MachineError::InvalidAddress),
        }
    }

    fn interpret(&mut self) -> Result<u64, MachineError> {
        while self.pc < self.code.len() as u64 {
            // TODO: Make sure we update pc after reading an instruction
            match self.read_insn(self.pc)? {
                Insn::LOAD64(i) => self.push(Insn::LOAD64(i)),
                Insn::RETURN => self.push(Insn::RETURN),
                Insn::BINOP(op) => {
                    // TODO: Match stack length >= 2 or raise machine error
                    let arg2 = self.pop();
                    let arg1 = self.pop();
                    // TODO: Fill out with all other codegen functions and figure out a clean way to handle this
                    let value = match op.as_str() {
                        // TODO: Put codegen into its own file somehow
                        "ADD" => codegen_add(arg1, arg2),
                        _ => MachineError::InvalidOperation,
                    };
                    self.push(value);
                }
                Insn::UNOP(op) => {
                    // TODO: Match stack length >= 1 or raise machine error
                    let arg = self.pop();
                    // TODO: Fill out with all other codegen functions and figure out a clean way to handle this
                    let value = match op.as_str() {
                        // TODO: Put codegen into its own file somehow
                        "UNOP_EXAMPLE" => codegen_unop_example(arg),
                        _ => MachineError::InvalidOperation,
                    };
                    self.push(value);
                }
            }
        }
        Ok(self.pc)
    }
}

// TODO: Add some mechanism to run from a file listing of bytecode
// TODO: Add some mechanism to run from STDIN of a bytecode listing
fn main() -> Result<(), MachineError> {
    println!("Hello, world!");
    // Read bytecode
    //
    // Instantiate machine
    // TODO: Replace the handjammed instructions with STDIN or file
    // TODO: Add box function into rust code too so we can generate better things here
    // let code = vec![Insn::LOAD64(5), Insn::RETURN];
    // Note: These following values are invalid, just for testing
    let code = vec![0u64, 1u64];
    let mut machine = Machine::new(code);
    //
    // Execute interpreter
    machine.interpret()?;
    Ok(())
}

// TODO: Add test harness here
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pop_fails() {
        let mut machine = Machine {
            pc: 0,
            code: vec![],
            stack: vec![],
        };
        let value = machine.pop();
        assert!(matches!(value, Err(MachineError::EmptyStackPop)));
    }

    #[test]
    fn test_read_end_of_code() {
        let mut machine = Machine {
            pc: 0,
            // TODO: Use real opcodes
            code: vec![0u64, 0u64],
            stack: vec![],
        };
        let value = machine.read_insn(2);
        assert!(matches!(value, Err(MachineError::InvalidAddress)));
    }

    // TODO: Add more machine error failure modes
}
// TODO: Add unbox tests
// TODO: Add interpreter tests
