import enum
import unittest
import struct
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional, assert_never
from collections.abc import Iterator

# TODO: Write DFS traversal for parser
# TODO: Add DFS tests
# TODO: Add box and unbox tests for all other types
# TODO: Add codegen
# TODO: Add codegen tests
# TODO: Write interpreter
# TODO: Add interpreter tests
# TODO: Redesign tokenizer and compiler to use Insns and their arity data. This should simplify the compiler function and make the parser more clear?
# TODO: Add test suite for duplicated code between compiler and interpreter. For now, this is just the box function

WHITESPACE = " \t\n"

# Mask for 64 bit values
PTR_MAX = (1 << 64) - 1


@dataclass
class Insn:
    bytecode: str
    scheme: Optional[str]
    mask: int
    tag: int
    shift: int
    arity: int

    def box(self, value: Optional[int] = None) -> bytes:
        return struct.pack("<Q", ((value or 0) << self.shift) + self.tag)

    def __repr__(self):
        return f"Insn:{self.bytecode}"


# Note: We assume all unaries begin with an alphabetical character in the tokenizer.
# If this assumption ever add unaries that do not fit this pattern, we will need to
# update the tokenizer. Future Yakob, watch out for this one.

# Note: We abuse the fact that all of these binary operations are single characters
# In our tokenizer, it searches for individual charactered binary operations.
# If we ever want to extend the syntax with a multi-character binary operation,
# we need to update the tokenizer. Consider yourself warned, future Yakob.

# Formatter is off for this section for easier reading
# The instructions are sorted by arity
# TODO: Add opcodes for booleans and characters
insns = [
    Insn(bytecode="LOAD64", scheme=None, mask=0b11, tag=0b11, shift=2, arity=0),
    Insn(
        bytecode="RETURN",
        scheme=None,
        mask=0xFFFF0000,
        tag=0x00020000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="ADD1",
        scheme="add1",
        mask=0xFFFF0000,
        tag=0x00120000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="SUB1",
        scheme="sub1",
        mask=0xFFFF0000,
        tag=0x00220000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="TOCHAR",
        scheme="integer->char",
        mask=0xFFFF0000,
        tag=0x00320000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="TOINT",
        scheme="char->integer",
        mask=0xFFFF0000,
        tag=0x00420000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="NULLPRED",
        scheme="null?",
        mask=0xFFFF0000,
        tag=0x00520000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="ZEROPRED",
        scheme="zero?",
        mask=0xFFFF0000,
        tag=0x00620000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="NOT", scheme="not", mask=0xFFFF0000, tag=0x00720000, shift=32, arity=1
    ),
    Insn(
        bytecode="INTPRED",
        scheme="integer?",
        mask=0xFFFF0000,
        tag=0x00820000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="BOOLPRED",
        scheme="boolean?",
        mask=0xFFFF0000,
        tag=0x00920000,
        shift=32,
        arity=1,
    ),
    Insn(
        bytecode="ADD", scheme="+", mask=0xFFFF0000, tag=0x00030000, shift=32, arity=2
    ),
    Insn(
        bytecode="MULT", scheme="*", mask=0xFFFF0000, tag=0x00130000, shift=32, arity=2
    ),
    Insn(
        bytecode="SUB", scheme="-", mask=0xFFFF0000, tag=0x00230000, shift=32, arity=2
    ),
    Insn(
        bytecode="LESS", scheme="<", mask=0xFFFF0000, tag=0x00330000, shift=32, arity=2
    ),
    Insn(
        bytecode="EQUAL", scheme="=", mask=0xFFFF0000, tag=0x00430000, shift=32, arity=2
    ),
]

insns_by_scheme = {i.scheme: i for i in insns if i.scheme is not None}
insns_by_bytecode = {i.bytecode: i for i in insns}


# -------- TOKENIZER DATA STRUCTURES --------
class Token(enum.IntEnum):
    NOP = enum.auto()  # No operation
    UNOP = enum.auto()  # Unary operation
    BINOP = enum.auto()  # Binary operation
    KEYWORD = enum.auto()  # Keyword
    PAREN = enum.auto()  # Paren (opening or closing)
    INTEGER = enum.auto()  # Integer value
    STRING = enum.auto()  # String value


# -------- PARSER DATA STRUCTURES --------
# The tokenizer ingests tokens and emits SchemeFunctions and SchemePrimitives
# The compiler ingests SchemeFunctions and SchemePrimitives and emits opcodes.
# The parsed scheme abstract syntax tree is composed of functions as the first arguments of an S-expression,
# and the primitives as the type for each argument in the S-expression.
@dataclass
class SchemeFunctionMeta:
    label: str
    arity: int


# fmt: off
class SchemeFunction(enum.Enum):
    ADD       = SchemeFunctionMeta(            "+", 2)
    MULT      = SchemeFunctionMeta(            "*", 2)
    SUB       = SchemeFunctionMeta(            "-", 2)
    LESS      = SchemeFunctionMeta(            "<", 2)
    EQUAL     = SchemeFunctionMeta(            "=", 2)
    ADD1      = SchemeFunctionMeta(         "add1", 1)
    SUB1      = SchemeFunctionMeta(         "sub1", 1)
    INTTOCHAR = SchemeFunctionMeta("integer->char", 1)
    CHARTOINT = SchemeFunctionMeta("char->integer", 1)
    NULLPRED  = SchemeFunctionMeta(        "null?", 1)
    ZEROPRED  = SchemeFunctionMeta(        "zero?", 1)
    NOT       = SchemeFunctionMeta(          "not", 1)
    INTPRED   = SchemeFunctionMeta(     "integer?", 1)
    BOOLPRED  = SchemeFunctionMeta(     "boolean?", 1)
# fmt: on


# Valid primitive types are ["BOOLEAN", "INTEGER", "CHAR"]
@dataclass
class SchemePrimitive:
    prim_type: str
    value: int


# -------- COMPILER DATA STRUCTURES --------
# The Opcode class hodls all valid interpreter instructions


# TODO: Figure out how we're going to handle variadic arity here
@dataclass
class OpcodeMeta:
    mask: int
    tag: int
    shift: int
    arity: int


# TODO: Redesign interpreter to use opcodes
# TODO: Redesign bytecode generation to use opcodes rather than Insns
# TODO: Remove insns and insns class
# TODO: Add control flow instructions like jumps
# TODO: Figure out if we want to augment our interpreter with flags of some kind (or some such control mechanism)
# TODO: Clean up tagging system
# TODO: Make sure bytecode generation for the interpreter works
# TODO: Implement compilation for scheme functions
# TODO: Add python comppiler tests
# fmt: off
class Opcode:
    ADD    = OpcodeMeta(mask=0xFFFF0000, tag=0x00010000, shift=32, arity=2)
    SUB    = OpcodeMeta(mask=0xFFFF0000, tag=0x00020000, shift=32, arity=2)
    RETURN = OpcodeMeta(mask=0xFFFF0000, tag=0x00030000, shift=32, arity=1)
    LESS   = OpcodeMeta(mask=0xFFFF0000, tag=0x00040000, shift=32, arity=2)
    EQUAL  = OpcodeMeta(mask=0xFFFF0000, tag=0x00050000, shift=32, arity=2)
    CJUMP  = OpcodeMeta(mask=0xFFFF0000, tag=0x00060000, shift=32, arity=1)
    JUMP   = OpcodeMeta(mask=0xFFFF0000, tag=0x00070000, shift=32, arity=0)
    LOAD64 = OpcodeMeta(mask=0x00000003, tag=0x00000003, shift= 2, arity=0)
# fmt: on


def get_scheme_func(label: str, arity: int):
    try:
        func = SchemeFunction(SchemeFunctionMeta(label, arity))
    except ValueError:
        raise ValueError(
            f"Invalud unary operation {label}. There is no corresponding SchemeFunction object of arity {arity}."
        )
    return func


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
            case c if c in map(
                lambda insn: insn.scheme,
                filter(lambda insn: insn.arity == 2, insns),
            ):
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
                if string in map(
                    lambda insn: insn.scheme,
                    filter(lambda insn: insn.arity == 1, insns),
                ):
                    return (Token.UNOP, string)
                else:
                    raise ValueError(
                        f"String types are not supported and {string} was not identified as an operator."
                    )
                    # return (Token.STRING, string)
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

    # TODO: Support booleans
    # TODO: Support chars
    # TODO: Add boolean tests
    # TODO: Add char tests
    def parse_expr(self, expr: List[tuple[Token, str]]) -> Iterator:
        if expr == []:
            return ""
        match expr.pop(0):
            case (Token.INTEGER, value):
                yield SchemePrimitive("INTEGER", int(value))
            case (Token.UNOP, op):
                yield get_scheme_func(op, 1)
                yield from self.parse_expr(expr)
            case (Token.BINOP, op):
                yield get_scheme_func(op, 2)
                yield from self.parse_expr(expr)
                yield from self.parse_expr(expr)
            case (Token.PAREN, "("):
                closing = (Token.PAREN, ")")
                if len(expr) == 0:
                    raise ValueError(
                        "Closing parenthesis expected but there are no more tokens.."
                    )
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
            case tok:
                raise ValueError(f"Unexpected token: '{tok}'")

    def parse(self) -> str | List[object]:
        self.tokenize()
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
    def __init__(self, ast):
        self.ast = ast
        self.code = []
        self.max_locals_count = 0

    # Each compiled output is tuple containing (Insn, value)
    def traverse_expr(self, expr):
        match expr:
            case [SchemePrimitive(prim_type=_, value=_) as prim, *_]:
                yield prim
            case [SchemeFunction() as func, *rest]:
                arity = func.value.arity
                label = func.value.label
                if arity is None:
                    raise NotImplementedError("Function {func} has variadic arity.")
                elif arity > len(rest):
                    raise ValueError(
                        f"Function {label} expects {arity} arguments, but only {len(rest)} were provided."
                    )
                # Handle each argument
                for i in range(arity):
                    yield from self.traverse_expr(rest[i:])
                yield func
            case []:
                raise ValueError("No expression to traverse.")
            case x:
                raise ValueError(
                    f"Unexpected expression for reordering before compilation: {x}"
                )

    # Collect all results of traverse_expr and add a return onto the end
    def traverse_function(self, expr):
        return list(self.traverse_expr(expr))

    def compile_scheme_function(self, scheme_func: SchemeFunction):
        match scheme_func:
            case SchemeFunction.ADD:
                raise NotImplementedError
            case SchemeFunction.MULT:
                raise NotImplementedError
            case SchemeFunction.SUB:
                raise NotImplementedError
            case SchemeFunction.LESS:
                raise NotImplementedError
            case SchemeFunction.EQUAL:
                raise NotImplementedError
            case SchemeFunction.ADD1:
                raise NotImplementedError
            case SchemeFunction.SUB1:
                raise NotImplementedError
            case SchemeFunction.INTTOCHAR:
                raise NotImplementedError
            case SchemeFunction.CHARTOINT:
                raise NotImplementedError
            case SchemeFunction.NULLPRED:
                raise NotImplementedError
            case SchemeFunction.ZEROPRED:
                raise NotImplementedError
            case SchemeFunction.NOT:
                raise NotImplementedError
            case SchemeFunction.INTPRED:
                raise NotImplementedError
            case SchemeFunction.BOOLPRED:
                raise NotImplementedError
            case _:
                assert_never(scheme_func)

    def compile(self):
        self.code = []
        for lexeme in self.traverse_function(self.ast):
            match lexeme:
                case SchemeFunction() as function:
                    self.code += self.compile_scheme_function(function)
                case SchemePrimitive() as primitive:
                    self.code.append(primitive)
                case x:
                    raise ValueError(f"Unknown scheme type {x}")

    # TODO: Improve the representation of bytecode to each be 64 bits
    # This means replacing all the opcodes with some shift like in the scheme paper
    # We can use a simple tag for "keyword" and these get indexed for each keyword
    # To make this easy to extract back into rust, we can have have a python generated module
    # with the right unbox information and such
    def write_to_stream(self, f):
        for insn, value in self.code:
            # TODO: Replace to_bytes with struct pack of some kind
            f.write(insn.box(value))


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

    # def test_tokenize_one_str(self):
    #     self.assertEqual(self._tokenize_one("asdf"), (Token.STRING, "asdf"))

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

    # This dummy test handles a typo. The instruction should be add1 for a unary function, but ADD1 is just a string.
    # We should probably give a warning for something like this
    # def test_tokenize_add1(self):
    #     self.assertEqual(
    #         self._tokenize("(ADD1 6)"),
    #         [
    #             (Token.PAREN, "("),
    #             (Token.STRING, "ADD1"),
    #             (Token.INTEGER, 6),
    #             (Token.PAREN, ")"),
    #         ],
    #     )


class ParseTests(unittest.TestCase):
    def _parse(self, source: str) -> object:
        p = Parser(source)
        p.tokenize()
        p.parse()
        return p.ast

    def test_parse_nothing(self):
        self.assertEqual(self._parse(""), "")

    def test_parse_fixnum(self):
        self.assertEqual(self._parse("42"), SchemePrimitive("INTEGER", 42))

    def test_parse_fixnum_with_whitespace(self):
        self.assertEqual(self._parse("     43"), SchemePrimitive("INTEGER", 43))

    def test_parse_fixnum_with_newline_whitespace(self):
        self.assertEqual(self._parse("\n\n5"), SchemePrimitive("INTEGER", 5))

    def test_parse_fixnum_big_number(self):
        self.assertEqual(
            self._parse(" 4325245623542352 "),
            SchemePrimitive("INTEGER", 4325245623542352),
        )

    def test_parse_unop(self):
        self.assertEqual(
            self._parse("(not 1)"),
            [
                SchemeFunction.NOT,
                SchemePrimitive("INTEGER", 1),
            ],
        )

    def test_parse_add1(self):
        self.assertEqual(
            self._parse("(add1 6)"),
            [
                SchemeFunction.ADD1,
                SchemePrimitive("INTEGER", 6),
            ],
        )

    def test_parse_zero(self):
        self.assertEqual(
            self._parse("(zero? 6)"),
            [
                SchemeFunction.ZEROPRED,
                SchemePrimitive("INTEGER", 6),
            ],
        )

    def test_nested_unops(self):
        self.assertEqual(
            self._parse("(add1 (sub1 (integer->char (char->integer (null? 0)))))"),
            [
                SchemeFunction.ADD1,
                [
                    SchemeFunction.SUB1,
                    [
                        SchemeFunction.INTTOCHAR,
                        [
                            SchemeFunction.CHARTOINT,
                            [SchemeFunction.NULLPRED, SchemePrimitive("INTEGER", 0)],
                        ],
                    ],
                ],
            ],
        )

    def test_parse_binop(self):
        self.assertEqual(
            self._parse("(* 5 6)"),
            [
                SchemeFunction.MULT,
                SchemePrimitive("INTEGER", 5),
                SchemePrimitive("INTEGER", 6),
            ],
        )

    def test_parse_binop_too_many_args(self):
        with self.assertRaises(ValueError):
            self._parse("(* 5 6 7)")

    def test_parse_layered_binops_and_parens(self):
        self.assertEqual(
            self._parse("(+ 1 (* ((2)) 3))"),
            [
                SchemeFunction.ADD,
                SchemePrimitive("INTEGER", 1),
                [
                    SchemeFunction.MULT,
                    [[SchemePrimitive("INTEGER", 2)]],
                    SchemePrimitive("INTEGER", 3),
                ],
            ],
        )

    def test_parse_balanced_parens_simple(self):
        self.assertEqual(self._parse("( 5 )"), [SchemePrimitive("INTEGER", 5)])

    def test_parse_open_parens(self):
        with self.assertRaises(ValueError):
            self._parse("(")

    def test_parse_dangling_close_parens(self):
        with self.assertRaises(ValueError):
            self._parse(")")


class TraversalTests(unittest.TestCase):
    def _traverse(self, source: str) -> object:
        p = Parser(source)
        p.tokenize()
        p.parse()
        c = Compiler(p.ast)
        return c.traverse_function(p.ast)

    def test_add_two_integers(self):
        self.assertEqual(
            self._traverse("(+ 1 2)"),
            [
                SchemePrimitive("INTEGER", 1),
                SchemePrimitive("INTEGER", 2),
                SchemeFunction.ADD,
            ],
        )

    def test_add1(self):
        self.assertEqual(
            self._traverse("(add1 6)"),
            [
                SchemePrimitive("INTEGER", 6),
                SchemeFunction.ADD1,
            ],
        )


class CompilerTests(unittest.TestCase):
    def _compile(self, source: str) -> object:
        p = Parser(source)
        p.tokenize()
        p.parse()
        c = Compiler(p.ast)
        c.compile()
        return c.code


# TODO: Add compiler test to make sure a return is added to the end


# TODO: Figure out how we can assert that there is a test for every type for boxing
# TODO: Figure out how to convert test function names to lowercase
class BoxTests(unittest.TestCase):
    # Apart from fixnum, each Insn is boxed without value and can be tested simply
    # Note: we do not strictly enforce boolean match of no-value Insn box.
    # It's not a problem but would be cleaner if this were more strongly specified
    def test_box_insns(self):
        for insn in insns:
            self.assertEqual(insn.box(), struct.pack("<Q", insn.tag))

    def test_box_fixnum_zero(self):
        self.assertEqual(insns_by_bytecode["LOAD64"].box(0), struct.pack("<Q", 0b11))


def compile_program():
    source = sys.stdin.read()
    program = scheme_parse(source)
    compiler = Compiler(program)
    compiler.compile()
    compiler.write_to_stream(sys.stdout)


if __name__ == "__main__":
    compile_program()
