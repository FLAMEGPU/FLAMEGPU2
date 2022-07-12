from __future__ import print_function, unicode_literals
import six
import sys
import ast
import os
import tokenize
import warnings
from six import StringIO


def interleave(inter, f, seq):
    """Call f on each item in seq, calling inter() in between.
    """
    seq = iter(seq)
    try:
        f(next(seq))
    except StopIteration:
        pass
    else:
        for x in seq:
            inter()
            f(x)

class CodeGenException(Exception):
    """ Generic exception for errors raised in code generation """
    pass

class CodeGenerator:
    """Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded. """

    # represents built in functions
    pythonbuiltins = ["abs", "float", "int"]
    
    # basic types
    basic_arg_types = ['float', 'int']
    
    # supported math constansts    
    mathconsts = {"pi": "M_PI",
                  "e": "M_E",
                  "inf": "INFINITY",
                  "nan": "NAN",
                  }
    
    # support for most numpy types except complex numbers and float>64bit
    numpytypes = {"byte": "char",
                  "ubyte": "unsigned char",
                  "short": "short",
                  "ushort": "unsigned short",
                  "intc": "int",
                  "uintc": "unsigned int",
                  "uint": "unisgned int",
                  "longlong": "long long",
                  "ulonglong": "unsigned long long",
                  "half": "half",       # cuda supported
                  "single": "float",
                  "double": "double",
                  "longdouble": "long double",
                  "bool_": "bool",
                  "bool8": "bool",
                  # sized aliases
                  "int_": "long",
                  "int8": "int8_t",
                  "int16": "int16_t",
                  "int32": "int32_t",
                  "int64": "int64_t",
                  "intp": "intptr_t",
                  "uint_": "long",
                  "uint8": "uint8_t",
                  "uint16": "uint16_t",
                  "uint32": "uint32_t",
                  "uint64": "uint64_t",
                  "uintp": "uintptr_t",
                  "float_": "float",
                  "float16": "half",
                  "float32": "float",
                  "float64": "double"
                  }
    
    # getVariableType and setVariableType functions are added dynamically    
    fgpu_funcs = [ "getID", "getStepCounter", "getThreadIndex" ]   
    fgpu_attrs = ["ALIVE", "DEAD"]
    fgpu_input_msg_funcs = ["getIndex"] 
    fgpu_output_msg_funcs = []
    fgpu_agent_out_msg_funcs = ["getID"]
    fgpu_env_funcs = ["containsProperty"] # TODO: Get macro property
    fgpu_rand_funcs = []
    fgpu_message_types = ["MessageNone", "MessageBruteForce", "MessageBucket", "MessageSpatial2D", "MessageSpatial3D", "MessageArray", "MessageArray2D", "MessageArray3D"]
    
    _fgpu_types = {"Float": "float",
                  "Double": "double",
                  "Int": "int",
                  "UInt": "unsigned int",
                  "Int8": "int_8",
                  "UInt8": "uint_8",
                  "Char": "char",
                  "UChar": "unsigned char",
                  "Int16": "int_16",
                  "UInt16": "uint_16",
                  "Int32": "int_32",
                  "UInt32": "uint_32",
                  "Int64": "int_64",
                  "UInt64": "uint_64"
                 }


    def __init__(self, tree, file = sys.stdout):
        """CodeGenerator(tree, file=sys.stdout) -> None.
         Print the source for tree to file."""
        self.f = file
        self.future_imports = []
        self._indent = 0
        # dict of locals used to determine if variable already exists in assignments
        self._locals = ["FLAMEGPU"]
        self._device_functions = []
        self._message_iterator_var = None           # default
        self._input_message_var = 'message_in'      # default
        self._output_message_var = 'message_out'    # default
        self.dispatch(tree)
        print("", file=self.f)
        self.f.flush()
        
                
    def _deviceVariableFunctionName(self, tree, py_func, permitted_prefixes, allow_lengths = True):
        """
        Gets the device function name by translating a typed Python version to a templated cpp version.
        Python functions looks like getVariableFloatArray6 and translate to getVariable<float, 6>
        This function will detect and test against a set of known types and also extract the Array length
        This function returns None if the string is invalid in format but only throws an error if the format is correct but the type is invalid.
        """
        cpp_func_name = ""
        # extract function name start
        for prefix in permitted_prefixes:
            if py_func.startswith(prefix):
                cpp_func_name = prefix
                py_func = py_func[len(prefix):]
                break # dont allow the else
        else:
            return None
        # check type and lengths
        if allow_lengths:
            #split to get type and Array Length (TODO: This could instead be looked up from the model description)     
            type_and_length = py_func.split("Array")
            if type_and_length[0] not in self._fgpu_types:
                self.RaiseError(tree, f"'{type_and_length[0]}' is not a valid FLAME GPU type")
            t = self._fgpu_types[type_and_length[0]]
            # generate template args
            if (len(type_and_length) == 1):
                cpp_func_name += f"<{t}>"
            elif (len(type_and_length) == 2):
                cpp_func_name += f"<{t}, {type_and_length[1]}>"
            else:
                return None
        else:
            if py_func not in self._fgpu_types:
                self.RaiseError(tree, f"'{py_func}' is not a valid FLAME GPU type")
            t = self._fgpu_types[py_func]
            cpp_func_name += f"<{t}>"
        # return    
        return cpp_func_name
        


    def fill(self, text = ""):
        "Indent a piece of text, according to the current indentation level"
        self.f.write("\n"+"    "*self._indent + text)

    def write(self, text):
        "Append a piece of text to the current line."
        self.f.write(six.text_type(text))

    def enter(self):
        "Print '{', and increase the indentation."
        self.write("{")
        self._indent += 1

    def leave(self):
        "Decrease the indentation level and Print '}'"
        self._indent -= 1
        self.fill("}")

    def dispatch(self, tree):
        "Dispatcher function, dispatching tree type T to method _T."
        if isinstance(tree, list):
            for t in tree:
                self.dispatch(t)
            return
        meth = getattr(self, "_"+tree.__class__.__name__)
        meth(tree)
        
    def RaiseWarning(self, tree, str):
        warnings.warn(f"Warning ({tree.lineno}, {tree.col_offset}): {str}")
        
    def RaiseError(self, tree, str):
        raise CodeGenException(f"Error ({tree.lineno}, {tree.col_offset}): {str}")


    ############### Cutsom Unparsing methods ###############
    # These are special versions of the ast unparsing      #
    # dispatch functions.                                  #
    ########################################################
    
    def dispatchFGPUFunctionArgs(self, tree):
        """
        Handles arguments for a FLAME GPU function. Arguments must have syntax of `message_in: MessageInType, message_out: MessageOutType`
        Type hinting is required to translate a type into a FLAME GPU Message type implementation
        """
        # reset the locals variable stack
        self._locals = ["FLAMEGPU"]
        if len(tree.args.args) != 2:
            self.RaiseError(tree, "Expected two FLAME GPU function arguments (input message and output message)")
        # input message
        if not tree.args.args[0].annotation:
            self.RaiseError(tree.args.args[0], "Message input requires a supported type annotation")
        if tree.args.args[0].annotation.id not in self.fgpu_message_types:
            self.RaiseError(tree.args.args[0], "Message input type annotation not a supported message type")
        self._input_message_var = tree.args.args[0].arg  # store the message input variable name
        self.write("flamegpu::")        # requires namespace
        self.dispatch(tree.args.args[0].annotation)
        self.write(", ")
        # output message
        if not tree.args.args[1].annotation:
            self.RaiseError(tree.args.args[1], "Message output requires a supported type annotation")
        if tree.args.args[1].annotation.id not in self.fgpu_message_types:
            self.RaiseError(tree.args.args[1], "Message output type annotation not a supported message type")
        self._output_message_var = tree.args.args[1].arg  # store the message output variable name
        self.write("flamegpu::")        # requires namespace
        self.dispatch(tree.args.args[1].annotation)
    
    def dispatchType(self, tree):
        """
        There is a limited set of types and formats of type description supported. Types can be either;
        1) A python built in type of int or float, or
        2) A subset of numpy types prefixed with either numpy or np. e.g. np.int16
        This function translates and a catches unsupported types but does not translate a function call (i.e. cast)
        """
        if isinstance(tree, ast.Name):
            if tree.id not in self.basic_arg_types:
                self.RaiseError(tree, "Not a supported type")
            self.write(tree.id)
        elif isinstance(tree, ast.Attribute):
            if not isinstance(tree.value, ast.Name) :
                self.RaiseError(tree, "Not a supported type")
            if not (tree.value.id == "numpy" or tree.value.id == "np"):
                self.RaiseError(tree, "Not a supported type")
            if tree.attr not in self.numpytypes:
                self.RaiseError(tree, "Not a supported numpy type")
            self.write(self.numpytypes[tree.attr])
        else:
            self.RaiseError(tree, "Not a supported type")
    
    def dispatchFGPUDeviceFunctionArgs(self, tree):
        """
        Handles arguments for a FLAME GPU device function. Arguments must use type hinting to be translated to cpp.
        """
        # reset the locals variable stack
        self._locals = ["FLAMEGPU"]
        # input message
        first = True
        annotation = None
        for arg in tree.args.args:
            # ensure that there is a type annotation
            if not arg.annotation:
                self.RaiseError(arg, "Device function argument requires type annotation")
            # comma if not first
            if not first:
                self.write(", ")
            self.dispatchType(arg.annotation)
            self.write(f" {arg.arg}")
            # add arg to local variable stack
            self._locals.append(arg.arg)
            first = False    
    
    def dispatchMessageLoop(self, tree):
        """
        This is a special case of a range based for loop in which iterator item returns a const referecne to the message.
        Any user specified message value can be used.
        """
        self.fill("for (const auto& ")
        self.dispatch(tree.target)
        self.write(" : FLAMEGPU->")
        self.dispatch(tree.iter)
        self.write(")")
        self._message_iterator_var = tree.target.id
        self.enter()
        self.dispatch(tree.body)
        self.leave()
        self._message_iterator_var = None
    
    def dispatchMemberFunction(self,t):
        """
        A very limited set of function calls to members are supported so these are fully evaluated here.
        Function calls permittred are;
         * FLAMEGPU.function - a supported function call. e.g. FLAMEGPU.getVariableFloat(). This will be translated into a typed Cpp call.
         * message_input.function - a call to the message input variable (the name of which is specified in the function definition)
         * message_output.function - a call to the message output variable (the name of which is specified in the function definition)
         * FLAMEGPU.environment.function - the only nested attribute type. This will be translated into a typed Cpp call.
         * math.function - Any function calls from python `math` are translated to calls raw function calls. E.g. `math.sin()` becomes `sin()`
         * numpy.type - Any numpy types are translated to static casts
        """
        # Environment
        if isinstance(t.value, ast.Attribute):
            # only nested attribute type is environment
            if not isinstance(t.value.value, ast.Name):
                self.RaiseError(t, "Unknown or unsupported nested attribute")
            # FLAMEGPU->environment
            if t.value.value.id == "FLAMEGPU" and t.value.attr == "environment":
                # check it is a supported environment function
                self.write("FLAMEGPU->environment.")
                if t.attr in self.fgpu_env_funcs: 
                    # proceed
                    self.write(t.attr)
                else: 
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, t.attr, ["getProperty"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in FLAMEGPU.environment object")
                    # proceed
                    self.write(py_func)  
            # FLAMEGPU->random
            elif t.value.value.id == "FLAMEGPU" and t.value.attr == "random":
                # check it is a supported random function
                self.write("FLAMEGPU->random.")
                if t.attr in self.fgpu_rand_funcs: 
                    # proceed
                    self.write(t.attr)
                else: 
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, t.attr, ["uniform", "normal", "logNormal"], allow_lengths=False)
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in FLAMEGPU.random object")
                    # proceed
                    self.write(py_func) 
            elif t.value.value.id == "FLAMEGPU" and t.value.attr == "agent_out":
                # check it is a supported agent_out function
                self.write("FLAMEGPU->agent_out.")
                if t.attr in self.fgpu_agent_out_msg_funcs: 
                    # proceed
                    self.write(t.attr)
                else: 
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, t.attr, ["setVariable"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in FLAMEGPU.agent_out object")
                    # proceed
                    self.write(py_func)
            else:
                self.RaiseError(t, f"Unknown or unsupported nested attribute in {t.value.value.id}")
        # FLAMEGPU singleton
        elif isinstance(t.value, ast.Name):
            if t.value.id == "FLAMEGPU":
                # check for legit FGPU function calls 
                self.write("FLAMEGPU->")
                if t.attr in self.fgpu_funcs:
                    # proceed
                    self.write(t.attr)
                else:
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, t.attr, ["getVariable", "setVariable"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in FLAMEGPU object")
                    # proceed
                    self.write(py_func)
                
            # message input arg
            elif self._message_iterator_var:
                if t.value.id == self._message_iterator_var:
                    self.write(f"{self._message_iterator_var}.")
                    # check for legit FGPU function calls and translate
                    if t.attr in self.fgpu_input_msg_funcs:     
                        # proceed
                        self.write(t.attr)
                    else:
                        # possible getter setter type function
                        py_func = self._deviceVariableFunctionName(t, t.attr, ["getVariable"])
                        if not py_func:
                            self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._message_iterator_var}' message input iterable object")
                        # proceed
                        self.write(py_func)
                        
                        
                        
            # message output arg
            elif t.value.id == self._output_message_var:
                # check for legit FGPU function calls and translate
                self.write("FLAMEGPU->message_out.")
                if t.attr in self.fgpu_output_msg_funcs: 
                    # proceed
                    self.write(t.attr)
                else:
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, t.attr, ["setVariable"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._output_message_var}' message output object")
                    # proceed
                    self.write(py_func)
                
                    
            
            # math functions (try them in raw function call format) or constants
            elif t.value.id == "math":
                self.write(t.attr)
            # numpy types
            elif t.value.id == "numpy" or t.value.id == "np":
                if t.attr in self.numpytypes:
                    self.write(f"static_cast<{self.numpytypes[t.attr]}>")
                else: 
                    self.RaiseError(t, f"Unsupported numpy type {t.attr}")
            else:
                self.RaiseError(t, f"Global '{t.value.id}' identifier not supported")
        else:
            self.RaiseError(t, "Unsupported function call syntax")
     
    
   
     

    ############### Unparsing methods ######################
    # There should be one method per concrete grammar type #
    # Constructors should be grouped by sum type. Ideally, #
    # this would follow the order in the grammar, but      #
    # currently doesn't.                                   #
    ########################################################

    def _Module(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Interactive(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Expression(self, tree):
        self.dispatch(tree.body)

    # stmt
    def _Expr(self, tree):
        """
        Same as a standard python expression but ends with semicolon
        """
        self.fill()
        self.dispatch(tree.value)
        self.write(";")

    def _NamedExpr(self, tree):
        """
        No such concept in C++. Standard assignment can be used in any location.
        """
        self.write("(")
        self.dispatch(tree.target)
        self.write(" = ")
        self.dispatch(tree.value)
        self.write(")")

    def _Import(self, t):
        self.RaiseError(t, "Importing of modules not supported")

    def _ImportFrom(self, t):
        self.RaiseError(t, "Importing of modules not supported")

    def _Assign(self, t):
        """
        Assignment will use the auto type to define a variable at first use else will perform standard assignment.
        Note: There is no ability to create `const` variables unless this is inferred from the assignment expression.
        Multiple assignment is supported by cpp but not in the translator neither is assignment to complex expressions which are valid python syntax.
        """
        if len(t.targets) > 1:
            self.RaiseError(t, "Assignment to multiple targets not supported")
        if not isinstance(t.targets[0], ast.Name):
            self.RaiseError(t, "Assignment to complex expressions not supported")
        self.fill()
        # check if target exists in locals
        if t.targets[0].id not in self._locals :
            self.write("auto ")
            self._locals.append(t.targets[0].id)
        self.dispatch(t.targets[0])
        self.write(" = ")
        self.dispatch(t.value)
        self.write(";")

    def _AugAssign(self, t):
        """
        Similar to assignment in terms of restrictions. E.g. Allow only single named variable assignments.
        Also requires the named variable to already exist in scope.
        """
        if not isinstance(t.target, ast.Name):
            self.RaiseError(t, "Augmented assignment to complex expressions not supported")
        # check if target exists in locals
        if t.target.id not in self._locals :
            self.RaiseError(t, "Augmented assignment not permitted on variables not already assigned previously")
        self.fill()
        self.dispatch(t.target)
        self.write(" "+self.binop[t.op.__class__.__name__]+"= ")
        self.dispatch(t.value)
        self.write(";")

    def _AnnAssign(self, t):
        if not isinstance(t.target, ast.Name):
            self.RaiseError(t, "Augmented assignment to complex expressions not supported")
        self.fill()
        self.dispatchType(t.annotation)
        self.write(" ")
        self.dispatch(t.target)
        if t.value:
            self.write(" = ")
            self.dispatch(t.value)
        self.write(";")

    def _Return(self, t):
        """
        Standard cpp like return with semicolon.
        """
        self.fill("return")
        if t.value:
            self.write(" ")
            self.dispatch(t.value)
        self.write(";")

    def _Pass(self, t):
        self.fill(";")

    def _Break(self, t):
        self.fill("break;")

    def _Continue(self, t):
        self.fill("continue;")

    def _Delete(self, t):
        self.RaiseError(t, "Deletion not supported")

    def _Assert(self, t):
        """
        cassert does exist but probably not required in FGPU functions and unclear if supported by jitfy
        """
        self.RaiseError(t, "Assert not supported")

    def _Exec(self, t):
        self.RaiseError(t, "Exec not supported")

    def _Print(self, t):
        """
        This is old school python printing so no need to support
        """
        self.RaiseError(t, "Print not supported")
        
    def _Global(self, t):
        self.RaiseError(t, "Use of 'global' not supported")

    def _Nonlocal(self, t):
        self.RaiseError(t, "Use of 'nonlocal' not supported")

    def _Await(self, t):
        self.RaiseError(t, "Await not supported")

    def _Yield(self, t):
        self.RaiseError(t, "Yield not supported")

    def _YieldFrom(self, t):
        self.RaiseError(t, "Yield from not supported")

    def _Raise(self, t):
        """
        Exceptions are obviously supported in cpp but not in CUDA device code
        """
        self.RaiseError(t, "Exception raising not supported")

    def _Try(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _TryExcept(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _TryFinally(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _ExceptHandler(self, t):
        self.RaiseError(t, "Exceptions not supported")

    def _ClassDef(self, t):
        self.RaiseError(t, "Class definitions not supported")

    def _FunctionDef(self, t):
        """
        Checks the decorators of the function definition much must be either 'flamegpu_agent_function' or 'flamegpu_device_function'.
        Each is then processed in a different way using a specific dispatcher.
        Function calls are actually checked and only permitted (or user defined) function calls are supported.
        """
        self.write("\n")
        # check decorators
        if len(t.decorator_list) != 1 or not isinstance(t.decorator_list[0], ast.Name):
            self.RaiseError(t, "Function definitions require a single FLAMEGPU decorator of either 'flamegpu_agent_function' or 'flamegpu_device_function'")       
        # FLAMEGPU_AGENT_FUNCTION
        if t.decorator_list[0].id == 'flamegpu_agent_function' :
            if getattr(t, "returns", False):
                self.RaiseWarning(t, "Function definition return type not supported on 'flamegpu_agent_function'")
            self.fill(f"FLAMEGPU_AGENT_FUNCTION({t.name}, ")
            self.dispatchFGPUFunctionArgs(t)
            self.write(")")
        # FLAMEGPU_DEVICE_FUNCTION
        elif t.decorator_list[0].id == 'flamegpu_device_function' :
            self.fill(f"FLAMEGPU_DEVICE_FUNCTION ")
            if t.returns:
                self.dispatchType(t.returns)
            else:
                self.write("void")
            self.write(f" {t.name}(")
            self.dispatchFGPUDeviceFunctionArgs(t)
            self.write(")")
            # add to list of defined functions that can be called
            self._device_functions.append(t.name)
        else:
            self.RaiseError(t, "Function definition uses an unsupported decorator. Must use either 'flamegpu_agent_function' or 'flamegpu_device_function'")
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _AsyncFunctionDef(self, t):
        self.RaiseError(t, "Async functions not supported")

    def _For(self, t):
        """
        Two type for for loop are supported. Either;
        1) Message for loop in which case the format requires a iterator using the named FLAMEGPU function argument of 'message_in'
        2) A range based for loop with 1 to 3 arguments which is converted into a c style loop
        """
        # if message loop then process differently
        if isinstance(t.iter, ast.Name):
            if t.iter.id == self._input_message_var:
                self.dispatchMessageLoop(t)
            else:
                self.RaiseError(t, "Range based for loops only support message iteration using 'message_in' iterator")
        # do not support for else
        elif t.orelse:
            self.RaiseError(t, "For else not supported")
        # allow calls but only to range function
        elif isinstance(t.iter, ast.Call):
            if isinstance(t.iter.func, ast.Name):
                # catch case of message_input with arguments (e.g. spatial messaging)
                if t.iter.func.id == self._input_message_var:
                    self.dispatchMessageLoop(t)
                # otherwise permit only range based for loops
                elif t.iter.func.id == "range":
                    # switch on different uses of range based on number of arguments
                    if len(t.iter.args) == 1:
                        self.fill(f"for (int ")
                        self.dispatch(t.target)
                        self.write("=0;")
                        self.dispatch(t.target)
                        self.write("<")
                        self.dispatch(t.iter.args[0])
                        self.write(";")
                        self.dispatch(t.target)
                        self.write("++)")
                    elif len(t.iter.args) == 2:
                        self.fill(f"for (int ")
                        self.dispatch(t.target)
                        self.write("=")
                        self.dispatch(t.iter.args[0])
                        self.write(";")
                        self.dispatch(t.target)
                        self.write("<")
                        self.dispatch(t.iter.args[1])
                        self.write(";")
                        self.dispatch(t.target)
                        self.write("++)")
                    elif len(t.iter.args) == 3:
                        self.fill(f"for (int ")
                        self.dispatch(t.target)
                        self.write("=")
                        self.dispatch(t.iter.args[0])
                        self.write(";")
                        self.dispatch(t.target)
                        self.write("<")
                        self.dispatch(t.iter.args[1])
                        self.write(";")
                        self.dispatch(t.target)
                        self.write("+=")
                        self.dispatch(t.iter.args[2])
                        self.write(")")
                    else:
                        self.RaiseError(t, "Range based for loops requires use of 'range' function with arguments and not keywords")
                    self.enter()
                    self.dispatch(t.body)
                    self.leave()
                else:
                    self.RaiseError(t, "Range based for loops only support calls to the 'range' function")
            else:
                self.RaiseError(t, "Range based for loops only support message iteration or use of 'range'")
        else:
            self.RaiseError(t, "Range based for loops only support message iteration or use of 'range'")

    def _AsyncFor(self, t):
        self.RaiseError(t, "Async for not supported")   

    def _If(self, t):
        """
        Fairly straightforward translation to if, else if, else format
        """
        self.fill("if (")
        self.dispatch(t.test)
        self.write(")")
        self.enter()
        self.dispatch(t.body)
        self.leave()
        # collapse nested ifs into equivalent elifs.
        while (t.orelse and len(t.orelse) == 1 and
               isinstance(t.orelse[0], ast.If)):
            t = t.orelse[0]
            self.fill("else if (")
            self.dispatch(t.test)
            self.write(")")
            self.enter()
            self.dispatch(t.body)
            self.leave()
        # final else
        if t.orelse:
            self.fill("else")
            self.enter()
            self.dispatch(t.orelse)
            self.leave()

    def _While(self, t):
        """
        Straightforward translation to c style while loop
        """
        self.fill("while (")
        self.dispatch(t.test)
        self.write(")")
        self.enter()
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.RaiseError(t, "While else not supported")

    def _With(self, t):
        self.RaiseError(t, "With not supported")

    def _AsyncWith(self, t):
        self.RaiseError(t, "Async with not supported")

    # expr
    def _Bytes(self, t):
        self.RaiseError(t, "Bytes function not supported")

    def _Str(self, tree):
        self.write(repr(tree.s))
        
    def _JoinedStr(self, t):
        self.RaiseError(t, "Joined strings not supported")

    def _FormattedValue(self, t):
        self.RaiseError(t, "Formatted strings not supported")

    def _fstring_JoinedStr(self, t, write):
        self.RaiseError(t, "F strings not supported")

    def _fstring_Str(self, t, write):
        self.RaiseError(t, "F strings not supported")

    def _fstring_Constant(self, t, write):
        self.RaiseError(t, "F strings not supported")

    def _fstring_FormattedValue(self, t, write):
        self.RaiseError(t, "F strings not supported")

    def _Name(self, t):
        """
        Everything ends up as a Name once it is an identifier
        """
        self.write(t.id)

    def _NameConstant(self, t):
        self.RaiseError(t, "NameConstant depreciated and not supported")

    def _Repr(self, t):
        self.RaiseError(t, "Repr not supported")
     
    def _Constant(self, t):
        """
        Restrict most types of constant except for numeric types and constant strings
        Picks up some obvious conversions such as None and Bools
        """
        value = t.value
        if isinstance(value, tuple):
            self.RaiseError(t, "Tuples not supported")
        if isinstance(value, dict):
            self.RaiseError(t, "Dictionaries not supported")
        if isinstance(value, list):
            self.RaiseError(t, "Lists not supported")
        elif value is Ellipsis: # instead of `...` for Py2 compatibility
            self.RaiseError(t, "Ellipsis not supported")
        elif isinstance(value, str): 
            self.write(f'"{value}"')
        elif isinstance(value, (bytes, bytearray)):  # reject bytes strings e.g. b'123' 
            self.RaiseError(t, "Byte strings not supported")
        elif isinstance(value, bool):
            if value:
                self.write("true")
            else:
                self.write("false")
        elif value == None:
            self.write(0)
        else:
            self.write(repr(value))

    def _Num(self, t):
        self.write(repr(t.n))

    def _List(self, t):
        self.RaiseError(t, "Lists not supported")

    def _ListComp(self, t):
        self.RaiseError(t, "List comprehension not supported")

    def _GeneratorExp(self, t):
        self.RaiseError(t, "Generator expressions not supported")

    def _SetComp(self, t):
        self.RaiseError(t, "Set comprehension not supported")

    def _DictComp(self, t):
        self.RaiseError(t, "Dictionary comprehension not supported")

    def _comprehension(self, t):
        self.RaiseError(t, "Comprehension not supported")

    def _IfExp(self, t):
        """
        Equivalent to a ternary operator
        """
        self.dispatch(t.test)
        self.write(" ? ")
        self.dispatch(t.body)
        self.write(" : ")
        self.dispatch(t.orelse)


    def _Set(self, t):
        self.RaiseError(t, "Sets not supported")

    def _Dict(self, t):
        self.RaiseError(t, "Dictionaries not supported")

    def _Tuple(self, t):
        self.RaiseError(t, "Tuples not supported")

    unop = {"Invert":"~", "Not": "!", "UAdd":"+", "USub":"-"}
    def _UnaryOp(self, t):
        """
        Translate to C equivalent opertaors
        """
        self.write("(")
        self.write(self.unop[t.op.__class__.__name__])
        self.dispatch(t.operand)
        self.write(")")

    binop = { "Add":"+", "Sub":"-", "Mult":"*", "MatMult":"@", "Div":"/", "Mod":"%",
                    "LShift":"<<", "RShift":">>", "BitOr":"|", "BitXor":"^", "BitAnd":"&",
                    "FloorDiv":"//", "Pow": "**"}
    def _BinOp(self, t):
        """
        Python style pow and floordiv are not supported so translate to a function call.
        No matrix mul support.
        """
        op_name = t.op.__class__.__name__
        # translate pow into function call (no float version)
        if op_name == "Pow":
            self.write("pow(")
            self.dispatch(t.left)
            self.write(", ")
            self.dispatch(t.right)
            self.write(")")
        # translate floor div into function call (no float version)
        elif op_name == "FloorDiv":
            self.write("floor(")
            self.dispatch(t.left)
            self.write("/")
            self.dispatch(t.right)
            self.write(")")
        elif op_name == "MatMult":
            self.RaiseError(t, "Matrix multiplier operator not supported")
        else:
            self.write("(")
            self.dispatch(t.left)
            self.write(" " + self.binop[op_name] + " ")
            self.dispatch(t.right)
            self.write(")")

    cmpops = {"Eq":"==", "NotEq":"!=", "Lt":"<", "LtE":"<=", "Gt":">", "GtE":">=",
                        "Is":"==", "IsNot":"!=", "In":"in", "NotIn":"not in"}
    def _Compare(self, t):
        self.dispatch(t.left)
        for o, e in zip(t.ops, t.comparators):
            # detect list ops
            if o.__class__.__name__ == "In" or o.__class__.__name__ == "NotIn":
                self.RaiseError(t, "In and NotIn operators not supported")
            self.write(" " + self.cmpops[o.__class__.__name__] + " ")
            self.dispatch(e)

    boolops = {ast.And: '&&', ast.Or: '||'}
    def _BoolOp(self, t):
        """
        Translate to logical and/or operators in C
        """
        self.write("(")
        s = " %s " % self.boolops[t.op.__class__]
        interleave(lambda: self.write(s), self.dispatch, t.values)
        self.write(")")
       
    def _Attribute(self,t):
        """
        A very limited set of attributes are supported so these are fully evaluated here. Other places where attribute type expressions may occur will also evaluate them fully rather than recursively call this function.
        Attributes supported are only;
         * FLAMEGPU.attribute - a supported attribute e.g. FLAMEGPU.ALIVE. This will be translated into a namespace member.
         * math.constant - Any supported math constants are translated to C definition versions
        """
        # Only a limited set of globals supported
        func_dict = None
        
        # FLAMEGPU singleton
        if isinstance(t.value, ast.Name):
            if t.value.id == "FLAMEGPU":
                if t.attr in self.fgpu_attrs:
                    # proceed
                    self.write("flamegpu::")
                    self.write(t.attr)
                else:
                    self.RaiseError(t, f"Attriobute '{t.attr}' does not exist in FLAMEGPU object")
            # math functions (try them in raw function call format) or constants
            elif t.value.id == "math":
                if t.attr in self.mathconsts:
                    self.write(self.mathconsts[t.attr])
                else:
                    self.RaiseError(t, f"Unsupported math constant '{t.attr}'")
            # numpy types
            elif t.value.id == "numpy" or t.value.id == "np":
                # not sure how a numpy attribute would be used without function call or type hint but translate anyway 
                if t.attr in self.numpytypes:
                    self.write(self.numpytypes[t.attr])
                else: 
                    self.RaiseError(t, f"Unsupported numpy type {t.attr}")
            else:
                self.RaiseError(t, f"Global '{t.value.id}' identifiers not supported")
        else:
            self.RaiseError(t, "Unsupported attribute")

    def _Call(self, t):
        """
        Some basic checks are undertaken on calls to ensure that the function being called is either a builtin or defined device function.
        A special dispatcher is required 
        """
        # check calls but let attributes check in their own dispatcher
        funcs = self._device_functions + self.pythonbuiltins + [self._input_message_var] # message_input variable is a valid function name as certain message types have arguments on iterator
        if isinstance(t.func, ast.Name):
            if (t.func.id not in funcs):
                self.RaiseWarning(t, "Function call is not a defined FLAME GPU device function or a supported python built in.")
            # dispatch even if warning raised
            self.dispatch(t.func)
        elif isinstance(t.func, ast.Lambda):
            self.dispatch(t.func) # not supported
        else:
            # special handler for dispatching member function calls
            # This would otherwise be an attribute
            self.dispatchMemberFunction(t.func)        
        self.write("(")
        comma = False
        for e in t.args:
            if comma: self.write(", ")
            else: comma = True
            self.dispatch(e)
        if len(t.keywords):
            self.RaiseWarning(t, "Keyword argument not supported. Ignored.")
        if sys.version_info[:2] < (3, 5):
            if t.starargs:
                self.RaiseWarning(t, "Starargs not supported. Ignored.")
            if t.kwargs:
                self.RaiseWarning(t, "Kwargs not supported. Ignored.")
        self.write(")")

    def _Subscript(self, t):
        """
        Arrays are not supported so no need for this but no harm
        """
        self.dispatch(t.value)
        self.write("[")
        self.dispatch(t.slice)
        self.write("]")

    def _Starred(self, t):
        self.RaiseError(t, "Starred values not supported")

    # slice
    def _Ellipsis(self, t):
        self.RaiseError(t, "Ellipsis values not supported")

    def _Index(self, t):
        self.RaiseError(t, "Index values not supported")

    def _Slice(self, t):
        self.RaiseError(t, "Slicing values not supported")

    def _ExtSlice(self, t):
        self.RaiseError(t, "ExtSlice values not supported")

    # argument
    def _arg(self, t):
        """
        Arguments should be processed by a custom dispatcher and it should not be possible to get here
        """
        self.RaiseError(t, "Arguments should already have been processed")

    # others
    def _arguments(self, t):
        """
        Arguments should be processed by a custom dispatcher and it should not be possible to get here
        """
        self.RaiseError(t, "Arguments should already have been processed")

    def _keyword(self, t):
        self.RaiseError(t, "Keywords are not supported")

    def _Lambda(self, t):
        self.RaiseError(t, "Lambda is not supported")

    def _alias(self, t):
        self.RaiseError(t, "Aliasing is not supported")

    def _withitem(self, t):
        self.RaiseError(t, "With not supported")
