import enum
import unittest

WHITESPACE = " \t\n"

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

    def  skip_whitespace(self):
        self.pos = self.scan_until(lambda c: c not in WHITESPACE)

def scheme_parse(source: str) -> object:
    return Parser(source).parse()

class Compiler:
    def __init__(self):
        self.code = []

    def compile(self, expr):
        raise NotImplementedError("compile")

    def write_to_stream(self, f):
        raise NotImplementedError("write_to_stream")

class Insn(enum.IntEnum):
    pass

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
