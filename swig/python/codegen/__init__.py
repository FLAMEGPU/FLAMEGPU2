# coding: utf-8
from io import StringIO
from .codegen import CodeGenerator
from .codegen import ModuleExtractor
from .codegen import CodeGenException
from typing import Callable, Union, _SpecialForm
import ast
import inspect
import os

def codegen(tree: ast.AST, source: str = "PythonString", bypass_line_directive = False) -> str:
    """
    Code generate function will translate a pure python FLAME GPU function from an ast tree
    bypass_line_directive will prevent injection of #line directive which is useful within test suite for checking expected outputs
    """
    v = StringIO()
    CodeGenerator(tree, file=v, source=source, bypass_line_directive=bypass_line_directive)
    return v.getvalue()

def translate(function: Union[str, Callable]) -> str:
    """
    Translates a pure python agent function into a cpp one!
    """
    if isinstance(function, str):
        # No need to strip the module as line numbers will be correct
        tree = ast.parse(function)
        return codegen(tree)
    elif isinstance(function, Callable):
        module = inspect.getmodule(function)
        # If no file attribute then its probably from a notebookc ell or some other dynamic (non file) location
        source = "DynamicPython"
        if (hasattr(module, "__file__")):
            source = os.path.basename(module.__file__)
        # Strip the module to create a copy of the original preserving only valid device functions and the agent function/condition name
        # Line number will be preserved and used by the code generator to inject original source file and line numbers using #line directive
        stripped_module = ModuleExtractor(module=module, agent_func_name=function.__name__).process()
        # parse and code generate
        tree = ast.parse(stripped_module)

        # Filter constants
        module_members = inspect.getmembers(module);
        # Requires python 3.10+
        module_annontations = inspect.get_annotations(module)
        prepend_c_source = ""
        # Find all annotated variables
        for key, val in module_annontations.items():
            if (isinstance(val, _SpecialForm) and val._name == "Final") or val.__name__ == "constant":
                # Locate the literal for that variable (Python will precompute anything e.g. math.sqrt(12.5))
                for mem in module_members:
                    if key == mem[0]:
                        prepend_c_source += f"constexpr auto {mem[0]} = {mem[1]};\n"
                        break
        return prepend_c_source + codegen(tree, source=source) # use the orginal modules source file name
    else:
        raise CodeGenException(f"Error: translate function requires either a source string or Callable")