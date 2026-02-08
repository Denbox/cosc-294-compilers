import enum
import unittest
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Any, Tuple

# TODO: Add parser tests
# TODO: Write DFS traversal for parser
# TODO: Add DFS tests
# TODO: Add box and unbox tests for all other types
# TODO: Add codegen
# TODO: Add codegen tests
# TODO: Write interpreter
# TODO: Add interpreter tests
# TODO: Add unary functiosn
# TODO: Add unary function tests
# TODO: Add binary operators
# TODO: Add binary operator tests

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
        return self.from_ptr(struct.unpack("<Q", value)[0] >> self.shift)

def identity(x):
    return x

ptr_types = {
    "fixnum":     PtrType(2, 0b00000011, 2, identity, identity),
    "bool":       PtrType(7, 0b00011111, 7, identity, identity),
    "char":       PtrType(8, 0b00001111, 8, identity, identity),
    "empty_list": PtrType(8, 0b00101111, 8, identity, identity),
}

class Token(enum.IntEnum):
    NONE      = enum.auto()
    OPERATION = enum.auto()
    INTEGER   = enum.auto()
    KEYWORD   = enum.auto()
    STRING    = enum.auto()
    PAREN     = enum.auto()


# NOTE: Scheme allows for all sorts of weird variable names. We require that they start with alphanumeric
# This is a deviation from expected behavior.
class Parser:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.length = len(source)
        self.tokens = []
        self.depth = 0

    # Each execution of the lex_one function increments self.pos by at least 1 and emits at most 1 token
    # Running lex_one iteratively will terminate - either lexing the input or yielding an error
    def lex_one(self) -> Tuple[Token, object]:
        match self.peek():
            case ' ' | '\t' | '\n' as c:
                self.pos += 1
                return (Token.NONE, c)
            case '(':
                self.pos += 1
                return (Token.PAREN, '(')
            case ')':
                self.pos += 1
                return (Token.PAREN, ')')
            case '+' | '-' | '*' | '<' | '=' as c:
                self.pos += 1
                return (Token.OPERATION, c)
            case c if c.isdigit():
                end = self.scan_until(lambda c: not c.isdigit())
                integer = int(self.source[self.pos:end], 10)
                self.pos = end
                return (Token.INTEGER, integer)
            case c if c.isalpha():
                # TODO: Add keyword support here
                # We denote string ends as write space barriers or going in and out of nesting
                # Everything else is fair game
                end = self.scan_until(lambda c: c in " \t\n()[]")
                string = self.source[self.pos:end]
                self.pos = end
                return (Token.STRING, string)
            case '':
                raise EOFError(f"Unexpected end of input.\nCurrent tokenization: {self.tokens}\nSource: {self.source}\nCurrent position: {self.pos}")
            case x:
                raise ValueError(f"Cannot lex '{x}'.\nCurrent tokenization: {self.tokens}\nSource: {self.source}\nCurrent position: {self.pos}")

    def lex(self) -> object:
        while self.pos < self.length:
            token_tuple = self.lex_one()
            if token_tuple[0] != Token.NONE:
                self.tokens.append(token_tuple)
        return self.tokens
        
    
    def parse(self) -> object:
        self.skip_whitespace()
        match self.peek():
            case '':
                raise EOFError("Unexpected end of input")
            case c if c.isdigit():
                return self.parse_number()
            case c if c.isalpha():
                return self.parse_string()
            case '(':
                raise NotImplementedError()
            case ')':
                raise NotImplementedError()
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

class TokenizationTests(unittest.TestCase):
    def _lex_one(self, source: str) -> object:
        return Parser(source).lex_one()
    
    def _lex(self, source: str) -> object:
        return Parser(source).lex()

    def test_lex_num(self):
        self.assertEqual(self._lex_one("43"), (Token.INTEGER, 43))

    def test_lex_one_tab(self):
        self.assertEqual(self._lex_one("\t"), (Token.NONE, "\t"))

    def test_lex_one_space(self):
        self.assertEqual(self._lex_one(" "), (Token.NONE, " "))
        
    def test_lex_one_str(self):
        self.assertEqual(self._lex_one("asdf"), (Token.STRING, "asdf"))

    def test_lex_fixnum(self):
        self.assertEqual(self._lex("42"), [(Token.INTEGER, 42)])

    def test_lex_fixnum_with_whitespace(self):
        self.assertEqual(self._lex("     43"), [(Token.INTEGER, 43)])

    def test_lex_fixnum_with_newline_whitespace(self):
        self.assertEqual(self._lex("\n\n5"), [(Token.INTEGER, 5)])

    def test_lex_fixnum_big_number(self):
        self.assertEqual(self._lex(" 4325245623542352 "), [(Token.INTEGER, 4325245623542352)])

    def test_lex_multiple_fixnums(self):
        self.assertEqual(self._lex("5 6 7 7\n\t\t 7 "), [(Token.INTEGER, 5), (Token.INTEGER, 6), (Token.INTEGER, 7), (Token.INTEGER, 7), (Token.INTEGER, 7)])

    def test_lex_parens(self):
        self.assertEqual(self._lex("()"), [(Token.PAREN, "("), (Token.PAREN, ")")])

    def test_lex_open_paren(self):
        self.assertEqual(self._lex("("), [(Token.PAREN, "(")])

    def test_fixnums_in_parens(self):
        self.assertEqual(self._lex("(5(69) 23 5)"), [(Token.PAREN, "("), (Token.INTEGER, 5), (Token.PAREN, "("), (Token.INTEGER, 69), (Token.PAREN, ")"), (Token.INTEGER, 23), (Token.INTEGER, 5), (Token.PAREN, ")")])

# class ParseTests(unittest.TestCase):
#     def _lex(self, source: str) -> object:
#         return Parser(source).lex()

#     def test_lex_fixnum(self):
#         self.assertEqual(self._parse("42"), 42)

#     def test_lex_fixnum_with_whitespace(self):
#         self.assertEqual(self._parse("     43"), 43)

#     def test_lex_fixnum_with_newline_whitespace(self):
#         self.assertEqual(self._parse("\n\n5"), 5)
#     def test_lex_fixnum_big_number(self):
#         self.assertEqual(self._parse(" 4325245623542352 "), 4325245623542352)

class BoxTests(unittest.TestCase):
    def test_box_fixnum(self):
        self.assertEqual(ptr_types["fixnum"].box(5), struct.pack("<Q", 0b10111))

    def test_unbox_fixnum(self):
        self.assertEqual(ptr_types["fixnum"].unbox(struct.pack("<Q", 0b10111)), 5)

    def test_box_fixnum_roundtrip(self):
        self.assertEqual(ptr_types["fixnum"].unbox(ptr_types["fixnum"].box(29)), 29)

def compile_program():
    source = sys.stdin.read()
    program = scheme_parse(source)
    compiler = Compiler()
    compiler.compile_function(program)
    compiler.write_to_stream(sys.stdout)

if __name__ == "__main__":
    compile_program()
