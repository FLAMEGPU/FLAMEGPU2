# coding: utf-8
from io import StringIO
from .codegen import CodeGenerator
from .codegen import ModuleExtractor
from .codegen import CodeGenException
from typing import Callable, Union, _SpecialForm
import ast
import inspect
import os

def codegen(tree: ast.AST, source: str = "PythonString", bypass_line_directive: bool = False, source_line_offset: int = 0) -> str:
    """
    Code generate function will translate a pure python FLAME GPU function from an ast tree
    bypass_line_directive will prevent injection of #line directive which is useful within test suite for checking expected outputs
    source_line_offset is used to offset the line directive values by a given amount (useful if source has been prepended to agent function)
    """
    v = StringIO()
    CodeGenerator(tree, file=v, source=source, bypass_line_directive=bypass_line_directive, source_line_offset=source_line_offset)
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

        # Iterate the modules functions to obtain any agent ot device functions relevant for compilation
        module_functions = inspect.getmembers(module, inspect.isfunction)
        # check that the agent function is globally defined (non globals not supported)
        if not function in list(zip(*module_functions))[1]:         # zip list of tuples to get a list of names [0] and callables[1]
            raise CodeGenException(f"Error: Function passed to translate is not a global. Only globals are supported.")

        tree = None
        source = None
        source_line_offset = 0
        if (hasattr(module, "__file__") and module.__file__ != ""):   # Source os from a file so strip using the ModuleExtractor
            source = os.path.basename(module.__file__)
            # Strip the module to create a copy of the original preserving only valid device functions and the agent function/condition name
            # Line number will be preserved and used by the code generator to inject original source file and line numbers using #line directive
            stripped_module = ModuleExtractor(module=module, agent_func_name=function.__name__).process()
            # parse and code generate
            tree = ast.parse(stripped_module)
        else :
            # If no file attribute then its probably from a notebook cell or some other dynamic (non file) location
            source = "DynamicPython"
            prepend_source = ""
            # filter function by device function (modules functions are a tuple of (name: str, func: Callable))
            d_functions = [x[1] for x in module_functions if hasattr(x[1], '__is_pyflamegpu_device_function')]
            # get source for each function
            for d_f in d_functions:
                prepend_source += inspect.getsource(d_f);
            # get source for function and prepend device functions
            function_source = prepend_source + inspect.getsource(function)
            # get the starting line of the actual agent function to use in compilation errors (prepended source errors will be line 1)
            source_line_offset = -(prepend_source.count('\n'))
            # parse the dynamic source file
            tree = ast.parse(function_source)

        # Filter constants and generate constexpr representations for prepending to C++ source
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
        return prepend_c_source + codegen(tree, source=source, source_line_offset=source_line_offset) # use the orginal modules source file name
    else:
        raise CodeGenException(f"Error: translate function requires either a source string or Callable")