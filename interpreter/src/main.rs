use std::fmt;

mod bytecode;
mod codegen;
use crate::bytecode::Insn;

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
            Some(bits) => bytecode::unbox(*bits).ok_or(MachineError::InvalidOpcode),
            None => Err(MachineError::InvalidAddress),
        }
    }

    fn interpret(&mut self) -> Result<u64, MachineError> {
        while self.pc < self.code.len() as u64 {
            // TODO: Make sure we update pc after reading an instruction
            match self.read_insn(self.pc)? {
                Insn::LOAD64(i) => self.push(Insn::LOAD64(i)),
                Insn::RETURN => self.push(Insn::RETURN),
                Insn::ADD => {
                    // TODO: Match stack length >= 2 or raise machine error
                    let arg2 = self.pop();
                    let arg1 = self.pop();
                    let value = codegen::add(arg1, arg2);
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

pub fn add(x: Insn, y: Insn) -> Result<Insn, MachineError> {
    if x == Insn::LOAD64(x_val) && y == Insn::LOAD64(y_val) {
        Ok(Insn::LOAD64(x_val + y_val))
    } else {
        Err(MachineError::InvalidOperation)
    }
}

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
