use std::fmt;

#[derive(Copy, Debug, Clone)]
enum Insn {
    LOAD64(u64),
    RETURN,
}

// TODO: Fix Error type to be more specific
// TODO: Add error type with ok_or_else into interpreter
#[derive(Debug, Clone)]
enum MachineError {
    EmptyStackRead(String),
    EmptyStackPop(String),
    InvalidAddress(String),
}

impl fmt::Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid first item to double")
    }
}

struct Machine {
    pc: u64, // 64 bits is far more than necessary but it's okay
    code: Vec<Insn>,
    stack: Vec<Insn>,
}

impl Machine {
    pub fn new(code: Vec<Insn>) -> Self {
        Self { pc: 0, code, stack: vec![] }
    }

    fn push(&mut self, i: Insn) {
        self.stack.push(i);
    }

    fn pop(&mut self) -> Result<Insn, MachineError> {
        match self.stack.pop() {
            Some(insn) => { Ok(insn) }
            None => { Err(MachineError::EmptyStackPop("Cannot pop from empty stack".to_string())) }
        }
    }

    fn read_word(&mut self) -> Result<Insn, MachineError> {
        match self.code.get(self.pc as usize) {
            Some(insn) => { Ok(*insn) }
            None => Err(MachineError::InvalidAddress(format!("Cannot read word with pc={} when the code length={}", self.pc, self.code.len())))
        }
    }

    fn interpret(&mut self) {
        while self.pc < self.code.len() as u64 {
            match self.read_word() {
                
            }
        }        
    }
}

// TODO: Add some mechanism to run from a file listing of bytecode
// TODO: Add some mechanism to run from STDIN of a bytecode listing
fn main() {
    println!("Hello, world!");
    // Read bytecode
    // 
    // Instantiate machine
    // TODO: Replace the handjammed instructions with STDIN or file
    let code = vec![Insn::LOAD64(5), Insn::RETURN];
    let machine = Machine::new(code);
    //
    // Execute interpreter
}

// TODO: Add test harness here
// TODO: Add unbox tests
// TODO: Add interpreter tests
