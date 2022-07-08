# coding: utf-8
from six.moves import cStringIO
from .codegen import CodeGenerator
from .codegen import CodeGenException
import ast

def codegen(tree: ast.AST):
    """
    Code generate function will tranlate a pure python FLAME GPU function from an ast tree
    """
    v = cStringIO()
    CodeGenerator(tree, file=v)
    return v.getvalue()

def translate(function_string: str):
    """
    Translates a pure python agent function into a cpp one!
    """
    tree = ast.parse(function_string)
    return codegen(tree)