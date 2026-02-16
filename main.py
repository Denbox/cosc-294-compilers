import enum
import unittest
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Any, Tuple, List
from collections.abc import Iterator

# TODO: Add parser tests
# TODO: Write DFS traversal for parser
# TODO: Add DFS tests
# TODO: Add box and unbox tests for all other types
# TODO: Add codegen
# TODO: Add codegen tests
# TODO: Write interpreter
# TODO: Add interpreter tests

WHITESPACE = " \t\n"

# Mask for 64 bit values
PTR_MAX = (1 << 64) - 1


@dataclass
class PtrType:
    mask: int
    tag: int
    shift: int
    # non-trivial function for non integer types
    to_int: Callable[[Any], int]
    from_ptr: Callable[[int], Any]

    def match_type(self, value: bytes) -> bool:
        return (struct.unpack("<Q", value)[0] & self.mask) == self.tag

    # Convert value to integer, then do the following
    # Left shift, remove overflow bits, xor tag (add works too)
    def box(self, value: Any) -> bytes:
        return struct.pack(
            "<Q", ((self.to_int(value) << self.shift) & PTR_MAX) + self.tag
        )

    # We assume the type has already been matched with `match_type`
    # unpack and unshift the proper amount, and convert value
    def unbox(self, value: bytes) -> Any:
        return self.from_ptr(struct.unpack("<Q", value)[0] >> self.shift)


def identity(x):
    return x


ptr_types = {
    "fixnum": PtrType(2, 0b00000011, 2, identity, identity),
    "bool": PtrType(7, 0b00011111, 7, identity, identity),
    "char": PtrType(8, 0b00001111, 8, identity, identity),
    "empty_list": PtrType(8, 0b00101111, 8, identity, identity),
}

# Note: We assume all unaries begin with an alphabetical character in the tokenizer.
# If this assumption ever add unaries that do not fit this pattern, we will need to
# update the tokenizer. Future Yakob, watch out for this one.
unops = [
    "add1",
    "sub1",
    "integer->char",
    "char->integer",
    "null?",
    "zero?",
    "not",
    "integer?",
    "boolean?",
]

# Note: We abuse the fact that all of these binary operations are single characters
# In our tokenizer, it searches for individual charactered binary operations.
# If we ever want to extend the syntax with a multi-character binary operation,
# we need to update the tokenizer. Consider yourself warned, future Yakob.
binops = [
    "+",
    "*",
    "-",
    "<",
    "=",
]


class Token(enum.IntEnum):
    NOP = enum.auto()  # No operation
    UNOP = enum.auto()  # Unary operation
    BINOP = enum.auto()  # Binary operation
    KEYWORD = enum.auto()  # Keyword
    PAREN = enum.auto()  # Paren (opening or closing)
    INTEGER = enum.auto()  # Integer value
    STRING = enum.auto()  # String value


# NOTE: Scheme allows for all sorts of weird variable names. We require that they start with alphanumeric
# This is a deviation from expected behavior.
class Parser:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.length = len(source)
        self.depth = 0
        self.tokens = []
        self.ast = []

    # Each execution of the tokenize_one function increments self.pos by at least 1 and emits at most 1 token
    # Running tokenize_one iteratively will terminate - either tokenizing the input or yielding an error
    def tokenize_one(self) -> Tuple[Token, object]:
        match self.peek():
            case c if c in WHITESPACE:
                self.pos += 1
                return (Token.NOP, c)
            case "(":
                self.pos += 1
                return (Token.PAREN, "(")
            case ")":
                self.pos += 1
                return (Token.PAREN, ")")
            case c if c in binops:
                self.pos += 1
                return (Token.BINOP, c)
            case c if c.isdigit():
                end = self.scan_until(lambda c: not c.isdigit())
                integer = int(self.source[self.pos : end], 10)
                self.pos = end
                return (Token.INTEGER, integer)
            case c if c.isalpha():
                end = self.scan_until(lambda c: c in " \t\n()[]")
                string = self.source[self.pos : end]
                self.pos = end
                if string in unops:
                    return (Token.UNOP, string)
                else:
                    return (Token.STRING, string)
            case "":
                raise EOFError(
                    f"Unexpected end of input.\nCurrent tokenization: {self.tokens}\nSource: {self.source}\nCurrent position: {self.pos}"
                )
            case x:
                raise ValueError(
                    f"Cannot tokenize '{x}'.\nCurrent tokenization: {self.tokens}\nSource: {self.source}\nCurrent position: {self.pos}"
                )

    def tokenize(self) -> List[object]:
        while self.pos < self.length:
            token_tuple = self.tokenize_one()
            if token_tuple[0] != Token.NOP:
                self.tokens.append(token_tuple)
        return self.tokens

    def parse_expr(self, expr: List[tuple[Token, object]]) -> Iterator:
        if expr == []:
            return ""
        match expr.pop(0):
            case (Token.INTEGER, _) as i:
                yield i
            case (Token.STRING, _) as s:
                yield s
            case (Token.UNOP, _) as u:
                yield u
                yield from self.parse_expr(expr)
            case (Token.PAREN, "("):
                closing = (Token.PAREN, ")")
                if len(expr) == 0:
                    raise ValueError("Closing parenthesis expected.")
                if expr[0] == closing:
                    expr.pop(0)
                else:
                    result = [*self.parse_expr(expr)]
                    if len(expr) == 0:
                        raise ValueError(
                            "Unexpected end of input. Closing parenthesis needed."
                        )
                    elif expr[0] != closing:
                        raise ValueError("Closing parenthesis expected.")
                    expr.pop(0)  # We need to remove the closing parenthesis
                    yield result
            case (Token.BINOP, _) as b:
                yield b
                yield from self.parse_expr(expr)
                yield from self.parse_expr(expr)
            case tok:
                raise ValueError(f"Unexpected token: '{tok}'")

    def parse(self) -> str | List[object]:
        tokens = self.tokens[:]
        # Use the default value of "" in case there are no tokens
        self.ast = next(self.parse_expr(tokens), "")
        if len(tokens) != 0:
            raise ValueError(
                f"Failed to parse all tokens.\nCurrent parse: {self.ast}\nRemaining Tokens: {tokens}"
            )
        return self.ast

    def peek(self, pos=None) -> str:
        if pos is None:
            pos = self.pos
        if pos >= self.length:
            return ""
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
            case bytes(x):
                for _, ptr_type in ptr_types.items():
                    if ptr_type.match_type(x):
                        # replace this with codegen, right now it's hand jammed to only work for fixnums
                        emit(Insn.LOAD64)
                        emit(ptr_type.box(expr))
                        return
                raise ValueError(f"No ptr_type matches tag for '{x}'".format(x))
            case x:
                raise ValueError(f"No code generation for '{x}'".format(x))

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
    def _tokenize_one(self, source: str) -> object:
        return Parser(source).tokenize_one()

    def _tokenize(self, source: str) -> object:
        return Parser(source).tokenize()

    def test_tokenize_num(self):
        self.assertEqual(self._tokenize_one("43"), (Token.INTEGER, 43))

    def test_tokenize_one_tab(self):
        self.assertEqual(self._tokenize_one("\t"), (Token.NOP, "\t"))

    def test_tokenize_one_space(self):
        self.assertEqual(self._tokenize_one(" "), (Token.NOP, " "))

    def test_tokenize_one_str(self):
        self.assertEqual(self._tokenize_one("asdf"), (Token.STRING, "asdf"))

    def test_tokenize_one_add1(self):
        self.assertEqual(self._tokenize_one("add1"), (Token.UNOP, "add1"))

    def test_tokenize_one_not(self):
        self.assertEqual(self._tokenize_one("not"), (Token.UNOP, "not"))

    def test_tokenize_one_char_to_int(self):
        self.assertEqual(
            self._tokenize_one("char->integer"), (Token.UNOP, "char->integer")
        )

    def test_tokenize_fixnum(self):
        self.assertEqual(self._tokenize("42"), [(Token.INTEGER, 42)])

    def test_tokenize_fixnum_with_whitespace(self):
        self.assertEqual(self._tokenize("     43"), [(Token.INTEGER, 43)])

    def test_tokenize_fixnum_with_newline_whitespace(self):
        self.assertEqual(self._tokenize("\n\n5"), [(Token.INTEGER, 5)])

    def test_tokenize_fixnum_big_number(self):
        self.assertEqual(
            self._tokenize(" 4325245623542352 "), [(Token.INTEGER, 4325245623542352)]
        )

    def test_tokenize_binops(self):
        self.assertEqual(
            self._tokenize("+*--=<"),
            [
                (Token.BINOP, "+"),
                (Token.BINOP, "*"),
                (Token.BINOP, "-"),
                (Token.BINOP, "-"),
                (Token.BINOP, "="),
                (Token.BINOP, "<"),
            ],
        )

    def test_tokenize_multiple_fixnums(self):
        self.assertEqual(
            self._tokenize("5 6 7 7\n\t\t 7 "),
            [
                (Token.INTEGER, 5),
                (Token.INTEGER, 6),
                (Token.INTEGER, 7),
                (Token.INTEGER, 7),
                (Token.INTEGER, 7),
            ],
        )

    def test_tokenize_parens(self):
        self.assertEqual(self._tokenize("()"), [(Token.PAREN, "("), (Token.PAREN, ")")])

    def test_tokenize_open_paren(self):
        self.assertEqual(self._tokenize("("), [(Token.PAREN, "(")])

    def test_fixnums_in_parens(self):
        self.assertEqual(
            self._tokenize("(5(69) 23 5)"),
            [
                (Token.PAREN, "("),
                (Token.INTEGER, 5),
                (Token.PAREN, "("),
                (Token.INTEGER, 69),
                (Token.PAREN, ")"),
                (Token.INTEGER, 23),
                (Token.INTEGER, 5),
                (Token.PAREN, ")"),
            ],
        )


class ParseTests(unittest.TestCase):
    def _parse(self, source: str) -> object:
        p = Parser(source)
        p.tokenize()
        p.parse()
        return p.ast

    def test_parse_nothing(self):
        self.assertEqual(self._parse(""), "")

    def test_parse_fixnum(self):
        self.assertEqual(self._parse("42"), (Token.INTEGER, 42))

    def test_parse_fixnum_with_whitespace(self):
        self.assertEqual(self._parse("     43"), (Token.INTEGER, 43))

    def test_parse_fixnum_with_newline_whitespace(self):
        self.assertEqual(self._parse("\n\n5"), (Token.INTEGER, 5))

    def test_parse_fixnum_big_number(self):
        self.assertEqual(
            self._parse(" 4325245623542352 "), (Token.INTEGER, 4325245623542352)
        )

    def test_parse_unop(self):
        self.assertEqual(
            self._parse("(not 1)"), [(Token.UNOP, "not"), (Token.INTEGER, 1)]
        )

    def test_nested_unops(self):
        print(self._parse("(add1 (sub1 (integer->char (char->integer (null? 0)))))"))
        self.assertEqual(
            self._parse("(add1 (sub1 (integer->char (char->integer (null? 0)))))"),
            [
                (Token.UNOP, "add1"),
                [
                    (Token.UNOP, "sub1"),
                    [
                        (Token.UNOP, "integer->char"),
                        [
                            (Token.UNOP, "char->integer"),
                            [(Token.UNOP, "null?"), (Token.INTEGER, 0)],
                        ],
                    ],
                ],
            ],
        )

    def test_parse_binop(self):
        self.assertEqual(
            self._parse("(* 5 6)"),
            [(Token.BINOP, "*"), (Token.INTEGER, 5), (Token.INTEGER, 6)],
        )

    def test_parse_binop_too_many_args(self):
        with self.assertRaises(ValueError):
            self._parse("(* 5 6 7)")

    def test_parse_layered_binops_and_parens(self):
        self.assertEqual(
            self._parse("(+ 1 (* ((2)) 3))"),
            [
                (Token.BINOP, "+"),
                (Token.INTEGER, 1),
                [(Token.BINOP, "*"), [[(Token.INTEGER, 2)]], (Token.INTEGER, 3)],
            ],
        )

    def test_parse_balanced_parens_simple(self):
        self.assertEqual(self._parse("( 5 )"), [(Token.INTEGER, 5)])

    def test_parse_open_parens(self):
        with self.assertRaises(ValueError):
            self._parse("(")

    def test_parse_dangling_close_parens(self):
        with self.assertRaises(ValueError):
            self._parse(")")


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
