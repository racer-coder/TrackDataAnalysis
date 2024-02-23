
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import numpy as np

from sly import Lexer, Parser
from sly.lex import LexError

op_map = {
    '+': np.add,
    '-': np.subtract,
    '*': np.multiply,
    '/': np.divide,
    '^': np.power,
    '<': np.less,
    '>': np.greater,
    '&': np.bitwise_and,
    '|': np.bitwise_or,
    '<=': np.less_equal,
    '>=': np.greater_equal,
    '==': np.equal,
    '!=': np.not_equal,
    'and': np.logical_and,
    'or': np.logical_or,
}

class ExprLex(Lexer):
    tokens = { VAR, UNIT, ID, INT, FLOAT, LTE, GTE, EQ, NEQ, AND, OR, NOT }
    literals = { '(', ')' }.union({k for k in op_map.keys() if len(k) == 1})

    ignore = ' \t\n'

    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    ID['and'] = AND
    ID['or'] = OR
    ID['not'] = NOT
    VAR = r"'[a-zA-z0-9_ ]+'"
    UNIT = r'\[[^\]]+\]'
    FLOAT = r'([.][0-9]+|[0-9]+[.][0-9]*)([eE][-+]?[0-9]+)?'
    INT = r'[0-9]+'
    LTE = r'<='
    GTE = r'>='
    EQ = r'=='
    NEQ = r'!='

class EvalLiteral:
    def __init__(self, lit):
        self._value = lit
        self.depends = set()

    def timecodes(self, log):
        return np.array([], dtype=np.int32)

    def values(self, log, timecodes):
        return self._value

class EvalReference:
    def __init__(self, name, unit):
        self._name = name
        self._unit = unit
        self.depends = {name}

    def timecodes(self, log):
        return log.get_channel_data(self._name, self._unit).timecodes

    def values(self, log, timecodes):
        return log.get_channel_data(self._name, self._unit).interp_many(timecodes)

class EvalOp:
    def __init__(self, op, *args):
        self._op = op
        self._args = args
        self.depends = set(d for a in args for d in a.depends)

    def timecodes(self, log):
        a = self._args[0].timecodes(log)
        for b in self._args[1:]:
            a = np.union1d(a, b.timecodes(log))
        return a

    def values(self, log, timecodes):
        return self._op(*[a.values(log, timecodes) for a in self._args])

class ParseError(Exception):
    def __init__(self, tok):
        if tok:
            self.args = ('Parse error at token', tok)
        else:
            self.args = ('Parse error at end of file',)
        self.token = tok

class ExprParse(Parser):
    tokens = ExprLex.tokens

    precedence = ( # python precedence, except ^ is power, not xor
        ('left', OR),
        ('left', AND),
        ('right', NOT),
        ('left', '<', '>', LTE, GTE, EQ, NEQ),
        ('left', '|'),
        ('left', '&'),
        ('left', '+', '-'),
        ('left', '*', '/'),
        ('right', 'UMINUS'),
        ('left', '^'),
        )

    @_('VAR')
    def expr(self, p):
        return EvalReference(p.VAR[1:-1], None)

    @_('VAR UNIT')
    def expr(self, p):
        return EvalReference(p.VAR[1:-1], p.UNIT[1:-1])

    @_('INT')
    def expr(self, p):
        return EvalLiteral(int(p.INT))

    @_('FLOAT')
    def expr(self, p):
        return EvalLiteral(float(p.FLOAT))

    @_('"(" expr ")"')
    def expr(self, p):
        return p.expr

    @_('"-" expr %prec UMINUS')
    def expr(self, p):
        return EvalOp(np.negative, p.expr)

    @_('NOT expr')
    def expr(self, p):
        return EvalOp(np.logical_not, p.expr)

    @_(*['expr "%s" expr' % k for k in op_map.keys() if len(k) == 1],
       *['expr %s expr' % k for k in ('LTE', 'GTE', 'EQ', 'NEQ', 'AND', 'OR')])
    def expr(self, p):
        return EvalOp(op_map[p[1]], p.expr0, p.expr1)

    #@_('ID "(" expr_list ")"')
    #def expr(self, p):
    #    return (p.ID, p.expr_list)
    #
    #@_('expr { "," expr }')
    #def expr_list(self, p):
    #    return [p.expr0] + p.expr1
    #
    #@_('')
    #def expr_list(self, p):
    #    return []

    def error(self, token):
        raise ParseError(token)

def compile(text):
    return ExprParse().parse(ExprLex().tokenize(text))
