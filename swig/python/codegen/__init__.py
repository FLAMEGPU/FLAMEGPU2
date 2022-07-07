# coding: utf-8
from six.moves import cStringIO
from .codegen import CodeGenerator
from .codegen import CodeGenException

__version__ = '1.0.0'

def codegen(tree):
    v = cStringIO()
    CodeGenerator(tree, file=v)
    return v.getvalue()

