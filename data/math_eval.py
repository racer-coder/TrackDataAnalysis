
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

# pylint: disable=used-before-assignment
# pylint: disable=undefined-variable
# pylint: disable=unsupported-assignment-operation
# pylint: disable=function-redefined

import importlib

import numpy as np

from sly import Lexer, Parser
from sly.lex import LexError # pylint: disable=unused-import

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

func_map = {
    ('if', 3): np.where,
    ('min', 2): np.minimum,
    ('max', 2): np.maximum,
}

class ExprLex(Lexer):
    tokens = { VAR, UNIT, ID, FLOAT, LTE, GTE, EQ, NEQ, AND, OR, NOT, COMMENT }
    literals = { '(', ')', ',', '.' }.union({k for k in op_map.keys() if len(k) == 1})

    ignore = ' \t\n'

    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    ID['and'] = AND
    ID['or'] = OR
    ID['not'] = NOT
    VAR = r"'[a-zA-z0-9_ ]+'"
    UNIT = r'\[[^\]]+\]'
    FLOAT = r'([.][0-9]+|[0-9]+([.][0-9]*)?)([eE][-+]?[0-9]+)?'
    LTE = r'<='
    GTE = r'>='
    EQ = r'=='
    NEQ = r'!='
    COMMENT = '#.*' # explicit token so we can color syntax highlight

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

class EvalUser(EvalOp):
    def values(self, log, timecodes):
        modname = 'usermathfunc.' + '.'.join(self._op[:-1])
        mod = importlib.import_module(modname)
        return mod.__dict__[self._op[-1]](*[a.values(log, timecodes) for a in self._args],
                                          timecodes=timecodes)

class EvalWrap(EvalOp):
    def __init__(self, arg):
        super().__init__(None, arg)

    def values(self, log, timecodes):
        val = self._args[0].values(log, timecodes)
        if not isinstance(val, np.ndarray):
            val = np.repeat(val, len(timecodes))
        return val

class ParseError(Exception):
    def __init__(self, msg, tok):
        if not isinstance(msg, tuple):
            msg = (msg,)
        self.args = msg
        self.token = tok

class ExprParse(Parser):
    tokens = ExprLex.tokens.difference({'COMMENT'})

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

    @_('ID { "." ID } "(" expr_list ")"')
    def expr(self, p):
        if p.ID1: # user function
            return EvalUser([p.ID0] + p.ID1, *p.expr_list)
        try:
            return EvalOp(func_map[(p.ID0, len(p.expr_list))], *p.expr_list)
        except KeyError:
            nargs = []
            for name, n in func_map.keys():
                if name == p.ID0:
                    nargs.append(n)
            if nargs:
                nargs.sort()
                raise ParseError("Function %s accepts %s arguments, not %d"
                                 % (p.ID0, ', '.join([str(x) for x in nargs]), len(p.expr_list)),
                                 p) # pylint: disable=raise-from-missing
            else:
                raise ParseError(("Unknown function", p.ID0), p) # pylint: disable=raise-from-missing

    @_('expr { "," expr }')
    def expr_list(self, p):
        return [p.expr0] + p.expr1

    @_('')
    def expr_list(self, p):
        return []

    def error(self, token):
        if not token:
            raise ParseError('Parse error at end of file', token)
        elif token.type != 'COMMENT':
            raise ParseError(('Parse error at token', token), token)
        else: # manually skip comments
            return next(self.tokens, None) # user manual says call errok() but that doesn't work...

def eat_comments(gen):
    for tok in gen:
        if tok.type != 'COMMENT':
            yield tok

def compile(text):
    return EvalWrap(ExprParse().parse(eat_comments(ExprLex().tokenize(text))))
