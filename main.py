import enum
import unittest
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Any

WHITESPACE = " \t\n"

# Mask for 64 bit values
PTR_MAX = (1<<64) - 1

@dataclass
class PtrType:
    mask: int
    tag: int
    shift: int
    # non-trivial function for non integer types
    to_int: Callable[Any, int]
    from_ptr: Callable[int, Any]

    def match_type(self, value: bytes) -> bool:
        return (struct.unpack("<Q", value) & self.mask) == self.tag
    
    # Convert value to integer, then do the following
    # Left shift, remove overflow bits, xor tag (add works too)
    def box(self, value: Any) -> bytes:
        return struct.pack("<Q", ((self.to_int(value) << self.shift) & PTR_MAX) + self.tag)

    # We assume the type has already been matched with `match_type`
    # unpack and unshift the proper amount, and convert value
    def unbox(self, value: bytes) -> Any:
        return self.from_ptr(struct.unpack("<Q", value) >> self.shift)

def identity(x):
    return x

ptr_types = {
    "fixnum": PtrType(3, 0, 2, identity, identity)
}


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.length = len(source)

    def parse(self) -> object:
        self.skip_whitespace()
        match self.peek():
            case '':
                raise EOFError("Unexpected end of input")
            case c if c.isdigit():
                return self.parse_number()
            case c:
                raise NotImplementedError(f"It's not a number silly it's a '{c}'")
        raise NotImplementedError("parse")

    def peek(self, pos=None) -> chr:
        if pos is None:
            pos = self.pos
        if pos >= self.length:
            return ''
        return self.source[pos]

    def eof(self, index=None) -> bool:
        if index is None:
            index = self.pos
        return index >= self.length

    def scan_until(self, cond) -> int:
        # If there is nothing left to scan, return
        if self.eof():
            return self.pos
        tmp_pos = self.pos
        while not (cond(self.peek(tmp_pos)) or self.eof(tmp_pos)):
            tmp_pos += 1
        # If we have reached this point, we have reached end of file or cond
        return tmp_pos
    
    def parse_number(self):
        # No error handling because the parser starts on a digit to call this method
        # and scan_until ensures the range is only digits
        end = self.scan_until(lambda c: not c.isdigit())
        num = int(self.source[self.pos:end], 10)
        self.pos = end
        return num

    def skip_whitespace(self):
        self.pos = self.scan_until(lambda c: c not in WHITESPACE)

def scheme_parse(source: str) -> object:
    return Parser(source).parse()

class Compiler:
    def __init__(self):
        self.code = []
        self.max_locals_count = 0

    def compile(self, expr):
        emit = self.code.append
        match expr:
            case int(x):
                for pt in ptr_types:
                    if pt.match_type(x):
                        # replace this with codegen, right now it's hand jammed to only work for fixnums
                        emit(Insn.LOAD64)
                        emit(pt.box(expr))
                        return
                raise ValueError(f"No ptr_type matches tag for '{x}'".format(x))
            case x:
                raise ValueError(f"No code generation for '{x}'".format(x))
        return

    def compile_function(self, expr):
        self.compile(expr)
        self.code.append(Insn.RETURN)

    def write_to_stream(self, f):
        for op in self.code:
            f.write(op.to_bytes(8, "little"))

class Insn(enum.IntEnum):
    LOAD64 = enum.auto()
    RETURN = enum.auto()


class ParseTests(unittest.TestCase):
    def _parse(self, source: str) -> object:
        return Parser(source).parse()

    def test_parse_fixnum(self):
        self.assertEqual(self._parse("42"), 42)

    def test_parse_fixnum_with_whitespace(self):
        self.assertEqual(self._parse("     43"), 43)

    def test_parse_fixnum_with_newline_whitespace(self):
        self.assertEqual(self._parse("\n\n5"), 5)

    def test_parse_fixnum_big_number(self):
        self.assertEqual(self._parse(" 4325245623542352 "), 4325245623542352)

def compile_program():
    source = sys.stdin.read()
    program = scheme_parse(source)
    compiler = Compiler()
    compiler.compile_function(program)
    compiler.write_to_stream(sys.stdout)

if __name__ == "__main__":
    compile_program()
