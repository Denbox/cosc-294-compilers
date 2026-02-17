use std::fmt;

#[derive(Copy, Debug, Clone)]
enum Insn {
    LOAD64(u64),
    RETURN,
}

// struct Pointer {

// }

// TODO: Fill this in and maybe fix the type
fn unbox(qword: u64) -> Result<u64, MachineError> {
    Ok(qword)
}

#[derive(Debug)]
enum MachineError {
    EmptyStackPop,
    InvalidAddress,
    // InvalidPointerType,
}

impl fmt::Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid first item to double")
    }
}

struct Machine {
    pc: u64, // 64 bits is far more than necessary but it's okay
    code: Vec<Insn>,
    stack: Vec<u64>,
}

impl Machine {
    pub fn new(code: Vec<Insn>) -> Self {
        Self {
            pc: 0,
            code,
            stack: vec![],
        }
    }

    fn push(&mut self, qword: u64) {
        self.stack.push(qword);
    }

    fn pop(&mut self) -> Result<u64, MachineError> {
        match self.stack.pop() {
            Some(qword) => Ok(qword),
            None => Err(MachineError::EmptyStackPop),
        }
    }

    fn read_insn(&mut self) -> Result<Insn, MachineError> {
        match self.code.get(self.pc as usize) {
            Some(insn) => Ok(*insn),
            None => Err(MachineError::InvalidAddress),
        }
    }

    fn interpret(&mut self) -> Result<u64, MachineError> {
        while self.pc < self.code.len() as u64 {
            match self.read_insn()? {
                Insn::LOAD64(qword) => {
                    // unbox and put onto the stack
                    self.push(unbox(qword)?);
                }
                Insn::RETURN => {}
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
    let code = vec![Insn::LOAD64(5), Insn::RETURN];
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
            pc: 2,
            code: vec![Insn::LOAD64(5), Insn::RETURN],
            stack: vec![],
        };
        let value = machine.read_insn();
        assert!(matches!(value, Err(MachineError::InvalidAddress)));
    }

    // TODO: Add more machine error failure modes
}
// TODO: Add unbox tests
// TODO: Add interpreter tests
