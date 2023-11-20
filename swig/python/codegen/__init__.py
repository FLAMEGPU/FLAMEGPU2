# coding: utf-8
from io import StringIO
from .codegen import CodeGenerator
from .codegen import CodeGenException
from typing import Callable, Union
import ast
import inspect

def codegen(tree: ast.AST) -> str:
    """
    Code generate function will translate a pure python FLAME GPU function from an ast tree
    """
    v = StringIO()
    CodeGenerator(tree, file=v)
    return v.getvalue()

def translate(function: Union[str, Callable]) -> str:
    """
    Translates a pure python agent function into a cpp one!
    """
    if isinstance(function, str):
        tree = ast.parse(function)
        return codegen(tree)
    elif isinstance(function, Callable):
        # If a Callable has been passed directly then we need to seek any functions with the 'pyflamegpu.device_function' decorator to include in the source for translation
        # There is no need to use unwrap on `function` as it uses `functools.wrap`
        # get the module of the function
        module = inspect.getmodule(function)
        # get functions
        module_functions = inspect.getmembers(module, inspect.isfunction)
        # check that the agent function is globally defined (non globals not supported)
        if not function in list(zip(*module_functions))[1]:         # zip list of tuples to get a list of names [0] and callables[1]
            raise CodeGenException(f"Error: Function passed to translate is not a global. Only globals are supported.")
        prepend_source = ""
        # filter function by device function (modules functions are a tuple of (name: str, func: Callable))
        d_functions = [x[1] for x in module_functions if hasattr(x[1], '__is_pyflamegpu_device_function')]
        # get source for each function
        for d_f in d_functions:
            prepend_source += inspect.getsource(d_f);
        # get source for function and prepend device functions
        function_source = prepend_source + inspect.getsource(function)
        tree = ast.parse(function_source)
        # Filter constants
        module_members = inspect.getmembers(module);
        # Emulate inspect.get_annotations() (requires python 3.10+)
        module_annontations = {}
        for mem in module_members:
            if mem[0] == "__annotations__":
                module_annontations = mem[1]
                break
        prepend_c_source = ""
        # Find all annotated variables
        for key, val in module_annontations.items():
            if val.__name__ == "Final" or val.__name__ == "constant":
                # Locate the literal for that variable (Python will precompute anything e.g. math.sqrt(12.5))
                for mem in module_members:
                    if key == mem[0]:
                        prepend_c_source += f"constexpr auto {mem[0]} = {mem[1]};\n"
                        break
        return prepend_c_source + codegen(tree)
    else:
        raise CodeGenException(f"Error: translate function requires either a source string or Callable")