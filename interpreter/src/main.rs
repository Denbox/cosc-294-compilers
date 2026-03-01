use std::fmt;

mod bytecode;
use crate::bytecode::Insn;

#[derive(Debug)]
enum MachineError {
    EmptyStackPop,
    InvalidAddress,
    InvalidOpcode,
    InvalidOperation,
    UnimplementedOpcode,
    NoReturn,
}

impl fmt::Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid first item to double")
    }
}

struct Machine {
    pc: u64, // 64 bits is far more than necessary but it's okay
    code: Vec<u64>,
    stack: Vec<bytecode::Insn>,
}

impl Machine {
    pub fn new(code: Vec<u64>) -> Self {
        Self {
            pc: 0,
            code,
            stack: vec![],
        }
    }

    fn push(&mut self, insn: bytecode::Insn) {
        self.stack.push(insn);
    }

    fn pop(&mut self) -> Result<bytecode::Insn, MachineError> {
        match self.stack.pop() {
            Some(insn) => Ok(insn),
            None => Err(MachineError::EmptyStackPop),
        }
    }

    fn read_insn(&mut self, index: u64) -> Result<bytecode::Insn, MachineError> {
        match self.code.get(index as usize) {
            Some(bits) => bytecode::unbox(*bits).ok_or(MachineError::InvalidOpcode),
            None => Err(MachineError::InvalidAddress),
        }
    }

    fn interpret(&mut self) -> Result<bytecode::Insn, MachineError> {
        while self.pc < self.code.len() as u64 {
            match self.read_insn(self.pc)? {
                bytecode::Insn::LOAD64(x) => self.push(bytecode::Insn::LOAD64(x)),
                bytecode::Insn::RETURN => {
                    let arg = self.pop()?;
                    return Ok(arg);
                }
                bytecode::Insn::ADD => {
                    // TODO: Match stack length >= 2 or raise machine error
                    let arg1 = self.pop()?;
                    let arg2 = self.pop()?;
                    let value = add(arg1, arg2)?;
                    self.push(value);
                }
                bytecode::Insn::ADD1 => {
                    let arg = self.pop()?;
                    let value = add1(arg)?;
                    self.push(value);
                }
                bytecode::Insn::SUB1 => {
                    let arg = self.pop()?;
                    let value = sub1(arg)?;
                    self.push(value);
                }
                _ => return Err(MachineError::UnimplementedOpcode),
            }
            self.pc += 1;
        }
        Err(MachineError::NoReturn)
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
    let result = machine.interpret()?;
    println!("{result:?}");
    Ok(())
}

fn add(x: Insn, y: Insn) -> Result<Insn, MachineError> {
    match (x, y) {
        (Insn::LOAD64(x_val), Insn::LOAD64(y_val)) => Ok(Insn::LOAD64(x_val + y_val)),
        _ => Err(MachineError::InvalidOperation),
    }
}

// TODO: Handle overflows
fn add1(x: Insn) -> Result<Insn, MachineError> {
    match x {
        Insn::LOAD64(val) => Ok(Insn::LOAD64(val + 1)),
        _ => Err(MachineError::InvalidOperation),
    }
}

// TODO: Handle underflows
fn sub1(x: Insn) -> Result<Insn, MachineError> {
    match x {
        Insn::LOAD64(val) => Ok(Insn::LOAD64(val - 1)),
        _ => Err(MachineError::InvalidOperation),
    }
}

// TODO: Choose better opcodes that the compiler generates. A lot of the codegen happens on the compiler side rather than the interpreter side. This is a redesign of sorts
#[cfg(test)]
mod interpreter_tests {
    use super::*;

    #[test]
    fn empty_pop_fails() {
        let mut machine = Machine {
            pc: 0,
            code: vec![],
            stack: vec![],
        };
        let value = machine.pop();
        assert!(matches!(value, Err(MachineError::EmptyStackPop)));
    }

    #[test]
    fn read_end_of_code() {
        let mut machine = Machine {
            pc: 0,
            // TODO: Use real opcodes
            code: vec![0u64, 0u64],
            stack: vec![],
        };
        let value = machine.read_insn(2);
        assert!(matches!(value, Err(MachineError::InvalidAddress)));
    }

    #[test]
    fn add_two_load64s() {
        let mut machine = Machine {
            pc: 0,
            code: vec![
                bytecode::box_insn(bytecode::Insn::LOAD64(1)),
                bytecode::box_insn(bytecode::Insn::LOAD64(2)),
                bytecode::box_insn(bytecode::Insn::ADD),
                bytecode::box_insn(bytecode::Insn::RETURN),
            ],
            stack: vec![],
        };
        let output = machine.interpret();
        assert!(matches!(output, Ok(bytecode::Insn::LOAD64(3))));
    }

    #[test]
    fn add1_load64() {
        let mut machine = Machine {
            pc: 0,
            code: vec![
                bytecode::box_insn(bytecode::Insn::LOAD64(1)),
                bytecode::box_insn(bytecode::Insn::ADD1),
                bytecode::box_insn(bytecode::Insn::RETURN),
            ],
            stack: vec![],
        };
        let output = machine.interpret();
        assert!(matches!(output, Ok(bytecode::Insn::LOAD64(2))));
    }

    #[test]
    fn sub1_load64() {
        let mut machine = Machine {
            pc: 0,
            code: vec![
                bytecode::box_insn(bytecode::Insn::LOAD64(12)),
                bytecode::box_insn(bytecode::Insn::SUB1),
                bytecode::box_insn(bytecode::Insn::RETURN),
            ],
            stack: vec![],
        };
        let output = machine.interpret();
        assert!(matches!(output, Ok(bytecode::Insn::LOAD64(11))));
    }
}

#[cfg(test)]
mod box_tests {
    use super::*;

    #[test]
    fn unbox_bytecode_to_load64() {
        assert!(matches!(
            bytecode::unbox(0x17),
            Some(bytecode::Insn::LOAD64(5))
        ));
    }

    #[test]
    fn box_load64_to_bytecode() {
        assert!(matches!(bytecode::box_insn(bytecode::Insn::LOAD64(16)), 67));
    }

    #[test]
    fn return_round_trip() {
        assert!(matches!(
            bytecode::unbox(bytecode::box_insn(bytecode::Insn::RETURN)),
            Some(Insn::RETURN)
        ))
    }

    // TODO: Figure out how to expand this into multiple tests
    // TODO: Fix this and figure out how to iterate over an enum
    // #[test]
    // fn unbox_roundtrips() {
    //     for insn in bytecode::Insn {
    //         match insn {
    //             bytecode::Insn(_) => continue,
    //             _ => assert!(matches!(bytecode::unbox(bytecode::box_insn(insn)), insn)),
    //         }
    //     }
    // }
}
// TODO: Add unbox tests
