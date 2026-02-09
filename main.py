import enum
import unittest
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Any, Tuple, List

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
    NOP     = enum.auto() # No operation
    UNOP    = enum.auto() # Unary operation
    BINOP   = enum.auto() # Binary operation
    KEYWORD = enum.auto() # Keyword
    PAREN   = enum.auto() # Paren (opening or closing)
    INTEGER = enum.auto() # Integer value
    STRING  = enum.auto() # String value

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
    # TODO: Add keyword support
    # TODO: Add parentheses support
    # TODO: Add unary operator support
    def tokenize_one(self) -> Tuple[Token, object]:
        match self.peek():
            case ' ' | '\t' | '\n' as c:
                self.pos += 1
                return (Token.NOP, c)
            case '(':
                self.pos += 1
                return (Token.PAREN, '(')
            case ')':
                self.pos += 1
                return (Token.PAREN, ')')
            # TODO: Fix some of these unaries
            case '+' | '-' | '*' | '<' | '=' as c:
                self.pos += 1
                return (Token.BINOP, c)
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
                raise ValueError(f"Cannot tokenize '{x}'.\nCurrent tokenization: {self.tokens}\nSource: {self.source}\nCurrent position: {self.pos}")

    def tokenize(self) -> List[object]:
        while self.pos < self.length:
            token_tuple = self.tokenize_one()
            if token_tuple[0] != Token.NOP:
                self.tokens.append(token_tuple)
        return self.tokens

    # Opcodes are ordered in reverse polish notation. This is because, when they are pushed to the stack, operations will be on top.
    def parse_expr(self, expr: List[Tuple(Token, object)]) -> [object]:
        output = []
        if expr == []:
            raise ValueError("Expected an expression.")
        tok = expr.pop(0)
        match tok:
            case (Token.PAREN, "("):
                self.depth += 1
                output.append(self.parse_expr(expr))
            case (Token.PAREN, ")"):
                self.depth -= 1
                if self.depth < 0:
                    raise ValueError("Unexpected closing parenthesis.")
                else:
                    # TODO: Update this
                    raise ValueError("Jacob is confused, what should happen here??")
            case (Token.UNOP, obj):
                output.append((Token.UNOP, obj))
                output.append(self.parse_expr(expr))
            case (Token.BINOP, obj):
                output.extend([(Token.BINOP, obj)])
                output.extend(self.parse_expr(expr))
                output.extend(self.parse_expr(expr))
            case (Token.INTEGER, x):
                output.extend([(Token.INTEGER, x)])
            case (Token.STRING, x):
                output.extend([(Token.STRING, x)])
            case x:
                raise ValueError(f"Unexpected token '{x}'.")
        return output

        # match expr:
        #     case []:
        #         yield
        #     case [(Token.UNOP, obj)]:
        #         yield (Token.UNOP, obj)
        #     case [(Token.INTEGER, obj)]:
        #         yield (Token.INTEGER, obj)
        #     case [(Token.STRING, obj)]:
        #         yield (Token.STRING, obj)
        #     case [(Token.BINOP, obj), x, y]:
        #         yield from self.parse_expr(x)
        #         yield from self.parse_expr(y)
        #         yield (Token.BINOP, obj)
        #     case [(Token.PAREN, '('), *middle, (Token.PAREN, ")")]:
        #         yield from self.parse_expr(middle)
        #     case [(Token.NOP, obj), *rest]:
        #         raise ValueError(f"We should never have a NOP in our expression parser. Token: {str(Token.NOP), obj}.")
        #     case [(Token.UNOP, obj), x, y, *rest]:
        #         raise ValueError(f"A unary operation should never have more than one argument, but token '{str(Token.UNOP), obj}' has '{x}' and '{y}' at least.")
        #     case [(Token.UNOP, obj)]:
        #         raise ValueError(f"A unary operation needs to have one argument, but token '{str(Token.UNOP), obj}' has none.")
        #     case [(Token.BINOP, obj), x, y, z, *rest]:
        #         raise ValueError(f"A binary operation should never have more than two arguments, but token '{str(Token.BINOP), obj}' has '{x}', '{y}', and {z} at least.")
        #     case [(Token.BINOP, obj)]:
        #         raise ValueError(f"A binary operation needs to have two arguments, but token '{str(Token.BINOP), obj}' has none.")
        #     case [(Token.BINOP, obj), x]:
        #         raise ValueError(f"A binary operation needs to have two arguments, but token '{str(Token.BINOP), obj}' has only '{x}'.")
        #     case [(Token.INTEGER, obj), x, *rest]:
        #         raise ValueError(f"An integer should never have arguments, but token '{str(Token.INTEGER), obj}' has at least '{x}'.")
        #     case [(Token.STRING, obj), x, *rest]:
        #         raise ValueError(f"A string should never have arguments, but token '{str(Token.STRING), obj}' has at least '{x}'.")
        #     case [(Token.PAREN, "("), *middle, x]:
        #         raise ValueError("Opening parenthesis must be matched by a closing parenthesis as the final term. Got '{x}' instead.")
        #     case [(Token.PAREN, "(")]:
        #         raise ValueError("Opening parenthesis cannot be the final token. It must be matched by a closing parenthesis.")
        #     case [(label, obj), *rest]:
        #         if not isinstance(label, Token):
        #             raise ValueError(f"Cannot parse non-token {label} with value {obj}. Remaining expression data: {rest}.")
        #         raise ValueError(f"Parsing is not supported for token: {label} with data {obj}")
        #     case [x, *rest]:
        #         raise ValueError(f"Parsing is not supported for non-token value: {x}. Remaining data: {rest}.")
                                    
    # TODO: Make the type of opcodes more specific  than object
    def parse(self) -> List[object]:
        # Add surrounding parentheses to get us into an expression for sure
        # Then we can always lop off this extra piece. This is super ugly and there should be something better to do
        self.ast = self.parse_expr([(Token.PAREN, "(")] + self.tokens + [(Token.PAREN, ")")])[0]
        return self.ast

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

    # TODO: Add a keyword test for each type of keyword
    def test_tokenize_one_kw(self):
        self.assertEqual(True, True)

    def test_tokenize_fixnum(self):
        self.assertEqual(self._tokenize("42"), [(Token.INTEGER, 42)])

    def test_tokenize_fixnum_with_whitespace(self):
        self.assertEqual(self._tokenize("     43"), [(Token.INTEGER, 43)])

    def test_tokenize_fixnum_with_newline_whitespace(self):
        self.assertEqual(self._tokenize("\n\n5"), [(Token.INTEGER, 5)])

    def test_tokenize_fixnum_big_number(self):
        self.assertEqual(self._tokenize(" 4325245623542352 "), [(Token.INTEGER, 4325245623542352)])

    def test_tokenize_multiple_fixnums(self):
        self.assertEqual(self._tokenize("5 6 7 7\n\t\t 7 "), [(Token.INTEGER, 5), (Token.INTEGER, 6), (Token.INTEGER, 7), (Token.INTEGER, 7), (Token.INTEGER, 7)])

    def test_tokenize_parens(self):
        self.assertEqual(self._tokenize("()"), [(Token.PAREN, "("), (Token.PAREN, ")")])

    def test_tokenize_open_paren(self):
        self.assertEqual(self._tokenize("("), [(Token.PAREN, "(")])

    def test_fixnums_in_parens(self):
        self.assertEqual(self._tokenize("(5(69) 23 5)"), [(Token.PAREN, "("), (Token.INTEGER, 5), (Token.PAREN, "("), (Token.INTEGER, 69), (Token.PAREN, ")"), (Token.INTEGER, 23), (Token.INTEGER, 5), (Token.PAREN, ")")])

    # TODO: Test tokenizeing of unary and binary operations

# TODO: Add unary operation expression parsing tests. Look for errors with multiple args, and no args. Ensure correct parsing with one arg.
# TODO: Same with binary operations
# TODO: Add tests for balanced parentheses
class ParseTests(unittest.TestCase):
    def _parse(self, source: str) -> object:
        p = Parser(source)
        p.tokenize()
        p.parse()
        return p.ast

    def test_parse_fixnum(self):
        self.assertEqual(self._parse("42"), (Token.INTEGER, 42))

    def test_parse_fixnum_with_whitespace(self):
        self.assertEqual(self._parse("     43"), (Token.INTEGER, 43))

    def test_parse_fixnum_with_newline_whitespace(self):
        self.assertEqual(self._parse("\n\n5"), (Token.INTEGER, 5))

    def test_parse_fixnum_big_number(self):
        self.assertEqual(self._parse(" 4325245623542352 "), (Token.INTEGER, 4325245623542352))
    
    def test_parse_balanced_parens_simple(self):
        self.assertEqual(self._parse("( 5 )"), [(Token.INTEGER, 5)])

    def test_parse_open_parens(self):
        with self.assertRaises(ValueError):
            self._parse("(")

    def test_parse_dangling_close_parens(self):
        with self.assertRaises(ValueError):
            self._parse(")")

    # TODO: Add and test unary operators
    # def test_parse_unary_op(self):
    #     self.assertEqual(self._parse(""))

    def test_parse_binary_op(self):
        self.assertEqual(self._parse("* 5 6"),[(Token.BINOP, '*'), (Token.INTEGER, 5), (Token.INTEGER, 6)] )

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
