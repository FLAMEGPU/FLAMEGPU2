from __future__ import print_function, unicode_literals
import sys
import ast
import os
import tokenize
import warnings
from io import StringIO

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
    fgpu_funcs = [ "getID", "getStepCounter", "getIndex", "isAgent", "isState" ]   
    fgpu_attrs = ["ALIVE", "DEAD"]
    fgpu_input_msg_funcs = ["radius", "at"]               # functions that can be called on message_in that do NOT return iterators
    fgpu_input_msg_iter_funcs = ["wrap", "vn", "vn_wrap"] # functions that can be called on message_in that do return iterators
    fgpu_input_msg_iter_var_funcs = ["getIndex", "getVirtualX", "getVirtualY", "getVirtualZ"] 
    fgpu_output_msg_funcs = ["setLocation", "setKey", "setIndex"]
    fgpu_agent_out_msg_funcs = ["getID"]
    fgpu_env_funcs = ["containsProperty", "containsMacroProperty", "getDirectedGraph"]
    fgpu_env_macro_funcs = ["exchange", "CAS", "min", "max"]
    fgpu_env_directed_graph_funcs = ["getVertexID", "getVertexIndex", "getEdgeSource", "getEdgeDestination", "getEdgeIndex"]
    fgpu_env_directed_graph_iter_funcs = ["outEdges", "inEdges"]
    fgpu_env_directed_graph_iter_in_var_funcs = ["getEdgeSource"]
    fgpu_env_directed_graph_iter_out_var_funcs = ["getEdgeDestination"]
    fgpu_rand_funcs = []
    fgpu_message_types = ["pyflamegpu.MessageNone", "pyflamegpu.MessageBruteForce", "pyflamegpu.MessageBucket", "pyflamegpu.MessageSpatial2D", "pyflamegpu.MessageSpatial3D", "pyflamegpu.MessageArray", "pyflamegpu.MessageArray2D", "pyflamegpu.MessageArray3D"]
    
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
                  "UInt64": "uint_64",
                  "ID": "flamegpu::id_t"
                 }


    def __init__(self, tree, file = sys.stdout):
        """CodeGenerator(tree, file=sys.stdout) -> None.
         Print the source for tree to file."""
        self.f = file
        self.future_imports = []
        self._indent = 0
        # dict of locals used to determine if variable already exists in assignments
        self._locals = {"pyflamegpu": 0}
        self._device_functions = []
        self._message_iterator_var = None             # default
        self._input_message_var = 'message_in'        # default
        self._output_message_var = 'message_out'      # default
        self._standalone_message_var = []             # default
        self._directed_graph_vars = []                # default
        self._directed_graph_in_iterator_var = None   # default
        self._directed_graph_out_iterator_var = None  # default
        self.dispatch(tree)
        print("", file=self.f)
        self.f.flush()
        
                
    def _deviceVariableFunctionName(self, tree, permitted_prefixes, allow_lengths = True):
        """
        Gets the device function name by translating a typed Python version to a templated cpp version.
        Python functions looks like getVariableFloatArray6 and translate to getVariable<float, 6>
        This function will detect and test against a set of known types and also extract the Array length
        This function returns None if the string is invalid in format but only throws an error if the format is correct but the type is invalid.
        """
        cpp_func_name = ""
        py_func = tree.attr
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
            #split to get type and Array Length (This could **potentially** be looked up from the model description but current syntax is consistent with swig bindings)     
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
        self.f.write(str(text))

    def enter(self):
        "Print '{', and increase the indentation."
        self.write("{")
        self._indent += 1

    def leave(self):
        "Decrease the indentation level and Print '}'"
        self._indent -= 1
        self.fill("}")
        # Purge _locals of out of scope variables
        d_key = [key for key, val in self._locals.items() if val > self._indent]
        for key in d_key:
            del self._locals[key]

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

    ############### Custom Unparsing methods ###############
    # These are special versions of the ast unparsing      #
    # dispatch functions.                                  #
    ########################################################
    
    def dispatchMacroEnvFunction(self, tree, tree_parent):
        """
        Function will handle a getMacroEnvironment function (assuming it is correctly formatted (by checking with _deviceVariableFunctionName first))
        """
        cpp_func_name = "getMacroProperty"
        py_func = tree.attr
        # extract type from function name
        py_type = py_func[len(cpp_func_name):]
        if py_type not in self._fgpu_types:
            self.RaiseError(tree, f"'{py_type}' is not a valid FLAME GPU type")
        # get cpp type
        t = self._fgpu_types[py_type]
        cpp_func_name += f"<{t}"
        # mess with the parent to extract (and remove arguments so they dont end up in the argument list)
        if not tree_parent.args :
            self.RaiseError(tree, f" Macro environment function '{py_func}' is expected to have some arguments.")
        # if more than one arg then the rest are bounds to translate
        if len(tree_parent.args) > 1:
            bounds = tree_parent.args[1:]
            # process bounds by appending to cpp function template arguments
            for i in bounds:
                if sys.version_info < (3,8,0) and isinstance(i, ast.Num): # num required for python 3.7
                    if not isinstance(i.n, int):
                        self.RaiseError(tree, f" Macro environment function argument '{i}' should be an integer value.")
                    cpp_func_name += f", {i.n}"
                else: # all Python > 3.7    
                    if not isinstance(i, ast.Constant):
                        self.RaiseError(tree, f" Macro environment function argument '{i}' should be an constant value (or Num in Python <3.8).")
                    if not isinstance(i.value, int):
                        self.RaiseError(tree, f" Macro environment function argument '{i}' should be an integer value.")
                    cpp_func_name += f", {i.value}"
            # remove bounds from argument list (in place)
            del tree_parent.args[1:]
        cpp_func_name += ">"
        self.write(cpp_func_name)

    def dispatchFGPUFunctionArgs(self, tree):
        """
        Handles arguments for a FLAME GPU function. Arguments must have syntax of `message_in: MessageInType, message_out: MessageOutType`
        Type hinting is required to translate a type into a FLAME GPU Message type implementation
        """
        # reset the locals variable stack
        self._locals = {"pyflamegpu": 0}
        if len(tree.args.args) != 2:
            self.RaiseError(tree, "Expected two FLAME GPU function arguments (input message and output message)")
        # input message
        if not tree.args.args[0].annotation:
            self.RaiseError(tree.args.args[0], "Message input requires a supported type annotation")
        if not isinstance(tree.args.args[0].annotation, ast.Attribute):
            self.RaiseError(tree.args.args[0], "Message input type annotation should be an attribute of the form pyflamegpu.MessageType")
        if not isinstance(tree.args.args[0].annotation.value, ast.Name):
            self.RaiseError(tree.args.args[0], "Message output type annotation should be an attribute of the form pyflamegpu.MessageType")
        input_message_attr = tree.args.args[0].annotation.value.id + "." + tree.args.args[0].annotation.attr
        if input_message_attr not in self.fgpu_message_types:
            self.RaiseError(tree.args.args[0], "Message input type annotation not a supported message type")
        self._input_message_var = tree.args.args[0].arg  # store the message input variable name
        self.write(f"flamegpu::{tree.args.args[0].annotation.attr}")        # requires namespace
        self.write(", ")
        # output message
        if not tree.args.args[1].annotation:
            self.RaiseError(tree.args.args[1], "Message output requires a supported type annotation")
        if not isinstance(tree.args.args[1].annotation, ast.Attribute):
            self.RaiseError(tree.args.args[1], "Message output type annotation should be an attribute of the form pyflamegpu.MessageType")
        if not isinstance(tree.args.args[1].annotation.value, ast.Name):
            self.RaiseError(tree.args.args[1], "Message output type annotation should be an attribute of the form pyflamegpu.MessageType")
        output_message_attr = tree.args.args[1].annotation.value.id + "." + tree.args.args[1].annotation.attr
        if output_message_attr not in self.fgpu_message_types:
            self.RaiseError(tree.args.args[1], "Message output type annotation not a supported message type")
        self._output_message_var = tree.args.args[1].arg  # store the message output variable name
        self.write(f"flamegpu::{tree.args.args[1].annotation.attr}")        # requires namespace
    
    def dispatchType(self, tree):
        """
        There is a limited set of types and formats of type description supported. Types can be either;
        1) A python built in type of int or float, or
        2) A subset of numpy types prefixed with either numpy or np. e.g. np.int16
        This function translates and a catches unsupported types but does not translate a function call (i.e. cast)
        """
        if isinstance(tree, ast.Name):
            if tree.id not in self.basic_arg_types:
                self.RaiseError(tree, "%s is not a supported type"%(tree.id))
            self.write(tree.id)
        elif isinstance(tree, ast.Attribute):
            if not isinstance(tree.value, ast.Name) :
                self.RaiseError(tree, "Not a supported type")
            if not (tree.value.id == "numpy" or tree.value.id == "np"):
                self.RaiseError(tree, "%s.%s is not a supported type"%(tree.value.id, tree.attr))
            if tree.attr not in self.numpytypes:
                self.RaiseError(tree, "%s.%s is not a supported numpy type"%(tree.value.id, tree.attr))
            self.write(self.numpytypes[tree.attr])
        else:
            self.RaiseError(tree, "Not a supported type")
    
    def dispatchFGPUDeviceFunctionArgs(self, tree):
        """
        Handles arguments for a FLAME GPU device function. Arguments must use type hinting to be translated to cpp.
        """
        # reset the locals variable stack
        self._locals = {"pyflamegpu": 0}
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
            self._locals[arg.arg] = self._indent
            first = False    
    
    def dispatchMessageIteratorCall(self, tree):
        """
        Message iterator call maybe a simple one (e.g. message_in(x, y, z)) or a call to a member (e.g. message_in.wrap())
        Using this function avoid using the global call one which may accept member function calls to things that are not iterators.
        """
        # simple case not a member function just an iterator with arguments
        if isinstance(tree.func, ast.Name):
            self.write(f"FLAMEGPU->{tree.func.id}")
        if isinstance(tree.func, ast.Attribute) :
            if isinstance(tree.func.value, ast.Name):
                # check that the iterator is supported
                if not tree.func.attr in self.fgpu_input_msg_iter_funcs:
                    self.RaiseError(tree, f"Message input loop iterator '{tree.func.attr}' is not supported.")
                self.write(f"FLAMEGPU->{tree.func.value.id}.{tree.func.attr}")
            else:
                self.RaiseError(tree, "Message input loop iterator format incorrect.")

        # handle function arguments        
        self.write("(")
        self._CallArguments(tree)
        self.write(")")

    def dispatchMessageLoop(self, tree):
        """
        This is a special case of a range based for loop in which iterator item returns a const reference to the message.
        Any user specified message value can be used.
        """
        self.fill("for (const auto& ")
        self.dispatch(tree.target)
        self.write(" : ")
        # if simple message iterator
        if isinstance(tree.iter, ast.Name):
            if not tree.iter.id == self._input_message_var:
                self.RaiseError(t, f"Message input loop requires use of '{self._input_message_var}' as iterator.")
            # write with prefix
            self.write(f"FLAMEGPU->{self._input_message_var}")
        # if it is a call then handle the different cases
        elif isinstance(tree.iter, ast.Call):
            self.dispatchMessageIteratorCall(tree.iter)
        #otherwise not supported
        else :
            self.RaiseError(tree, f"Message input loop iterator in unsupported format")
        self.write(")")
        self._message_iterator_var = tree.target.id
        self.enter()
        self.dispatch(tree.body)
        self.leave()
        self._message_iterator_var = None
    
    def dispatchGraphIteratorCall(self, tree):
        """
        Graph iterator call maybe be either (e.g. mygraph.edgesIn() or mygraph.edgesOut())
        Using this function avoid using the global call one which may accept member function calls to things that are not iterators.
        """
        # simple case not a member function just an iterator with arguments
        if isinstance(tree.func, ast.Attribute) :
            if isinstance(tree.func.value, ast.Name):
                # check that the iterator is supported
                if not tree.func.attr in self.fgpu_env_directed_graph_iter_funcs:
                    self.RaiseError(tree, f"Graph loop iterator '{tree.func.attr}' is not supported.")
                self.write(f"{tree.func.value.id}.{tree.func.attr}")
            else:
                self.RaiseError(tree, "Graph loop iterator format incorrect.")

        # handle function arguments        
        self.write("(")
        self._CallArguments(tree)
        self.write(")")        
        
    def dispatchGraphLoop(self, tree):
        """
        This is a special case of a range based for loop in which iterator item returns a const reference to the edge.
        Any user specified graph value can be used.
        """
        self.fill("for (const auto& ")
        self.dispatch(tree.target)
        self.write(" : ")
        # graph iterator only current has calls
        if isinstance(tree.iter, ast.Call):
            self.dispatchGraphIteratorCall(tree.iter)
        #otherwise not supported
        else :
            self.RaiseError(tree, f"Graph loop iterator in unsupported format")
        self.write(")")
        if tree.iter.func.attr == "inEdges":
            self._directed_graph_in_iterator_var = tree.target.id
        elif tree.iter.func.attr == "outEdges":
            self._directed_graph_out_iterator_var = tree.target.id
        self.enter()
        self.dispatch(tree.body)
        self.leave()
        self._directed_graph_in_iterator_var = None
        self._directed_graph_out_iterator_var = None
    
    def dispatchMemberFunction(self, t, t_parent):
        """
        A very limited set of function calls to members are supported so these are fully evaluated here.
        t_parent is the Call ast object required if the argument need to be modified (i.e. in the case of macro environment properties)
        Function calls permitted are;
         * pyflamegpu.function - a supported function call. e.g. pyflamegpu.getVariableFloat(). This will be translated into a typed Cpp call.
         * message_input.function - a call to the message input variable (the name of which is specified in the function definition)
         * msg.function - a call to the message input iterator objection variable (the name of which is specified in the message function loop)
         * message_output.function - a call to the message output variable (the name of which is specified in the function definition)
         * pyflamegpu.environment.function - the only nested attribute type. This will be translated into a typed Cpp call.
         * math.function - Any function calls from python `math` are translated to calls raw function calls. E.g. `math.sin()` becomes `sin()`
         * numpy.type - Any numpy types are translated to static casts
        """
        # it could be possible that the Call object has no value property e.g. a()()
        if not hasattr(t, "value"):
            self.RaiseError(t, f"Function call is in an unsupported format.")

        # Nested member functions (e.g. x.y.z())
        if isinstance(t.value, ast.Attribute):
            # store some information about the source of this function call in parent as this may be useful for validation in whatever has called this function
            t_parent.call_type = None
            # only nested attribute type is environment
            if not isinstance(t.value.value, ast.Name):
                self.RaiseError(t, "Unknown or unsupported nested attribute")
            # pyflamegpu.environment
            if t.value.value.id == "pyflamegpu" and t.value.attr == "environment":
                # check it is a supported environment function
                self.write("FLAMEGPU->environment.")
                if t.attr in self.fgpu_env_funcs: 
                    # proceed
                    self.write(t.attr)
                else: 
                    # simple getProperty type function
                    if t.attr.startswith('getProperty') :
                        # possible getter setter type function
                        py_func = self._deviceVariableFunctionName(t, ["getProperty"])
                        if not py_func:
                            self.RaiseError(t, f"Function '{t.attr}' is not a supported pyflamegpu.environment property function.")
                        # write the getProperty type function
                        self.write(py_func)
                        t_parent.call_type = "Environment"
                    # need to catch case of getMacroProperty as arguments need to be translated into template parameters in cpp (and py_func can be ignored)
                    elif t.attr.startswith("getMacroProperty"):
                        # possible getter setter type function (Note: getMacroProperty only supports a subset of types but type checking is not performed. This is best left to the compiler.)
                        # no not permit lengths (e.g. Float4) as these will be passed as arguments
                        py_func = self._deviceVariableFunctionName(t, ["getMacroProperty"], allow_lengths=False)
                        if not py_func:
                            self.RaiseError(t, f"Function '{t.attr}' is not a supported pyflamegpu.environment macro property function.")
                        # handle case
                        self.dispatchMacroEnvFunction(t, t_parent)
                        t_parent.call_type = "MacroEnvironment"
                    else:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in pyflamegpu.environment object")
                          
            # pyflamegpu.random
            elif t.value.value.id == "pyflamegpu" and t.value.attr == "random":
                # check it is a supported random function
                self.write("FLAMEGPU->random.")
                if t.attr in self.fgpu_rand_funcs: 
                    # proceed
                    self.write(t.attr)
                else: 
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, ["uniform", "normal", "logNormal"], allow_lengths=False)
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in pyflamegpu.random object")
                    # proceed
                    self.write(py_func) 
                    t_parent.call_type = "Random"
            elif t.value.value.id == "pyflamegpu" and t.value.attr == "agent_out":
                # check it is a supported agent_out function
                self.write("FLAMEGPU->agent_out.")
                if t.attr in self.fgpu_agent_out_msg_funcs: 
                    # proceed
                    self.write(t.attr)
                else: 
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, ["setVariable"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in pyflamegpu.agent_out object")
                    # proceed
                    self.write(py_func)
                    t_parent.call_type = "AgentOut"
            else:
                self.RaiseError(t, f"Unknown or unsupported nested attribute {t.value.attr} in {t.value.value.id}")
        # Non nested member functions (e.g. x.y())
        elif isinstance(t.value, ast.Name):
            # pyflamegpu singleton
            if t.value.id == "pyflamegpu":
                # check for legit FGPU function calls 
                self.write("FLAMEGPU->")
                if t.attr in self.fgpu_funcs:
                    # proceed
                    self.write(t.attr)
                else:
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, ["getVariable", "setVariable"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in pyflamegpu object")
                    # proceed
                    self.write(py_func)

            # message_in function using whatever variable was named in function declaration (e.g radius)
            elif t.value.id == self._input_message_var:
                # only process functions on message_in that are not iterators
                if t.attr in self.fgpu_input_msg_funcs:
                    self.write(f"FLAMEGPU->{self._input_message_var}.")
                    self.write(t.attr)  
                else:
                    self.RaiseError(t, f"Message input variable '{self._input_message_var}' does not have a supported function '{t.attr}'") 

            # message input iterator arg
            elif self._message_iterator_var and t.value.id == self._message_iterator_var:
                    self.write(f"{self._message_iterator_var}.")
                    # check for legit FGPU function calls and translate
                    if t.attr in self.fgpu_input_msg_iter_var_funcs:     
                        # proceed
                        self.write(t.attr)
                    else:
                        # possible getter setter type function
                        py_func = self._deviceVariableFunctionName(t, ["getVariable"])
                        if not py_func:
                            self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._message_iterator_var}' message input iterable object")
                        # proceed
                        self.write(py_func)

            # directed graph iterator arg
            elif self._directed_graph_in_iterator_var and t.value.id == self._directed_graph_in_iterator_var:
                    self.write(f"{self._directed_graph_in_iterator_var}.")
                    # check for legit FGPU function calls and translate
                    if t.attr in self.fgpu_env_directed_graph_iter_in_var_funcs:
                        # proceed
                        self.write(t.attr)
                    else:
                        # possible getter setter type function
                        py_func = self._deviceVariableFunctionName(t, ["getProperty"])
                        if not py_func:
                            self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._directed_graph_in_iterator_var}' graph iterable object")
                        # proceed
                        self.write(py_func)
            elif self._directed_graph_out_iterator_var and t.value.id == self._directed_graph_out_iterator_var:
                    self.write(f"{self._directed_graph_out_iterator_var}.")
                    # check for legit FGPU function calls and translate
                    if t.attr in self.fgpu_env_directed_graph_iter_out_var_funcs:
                        # proceed
                        self.write(t.attr)
                    else:
                        # possible getter setter type function
                        py_func = self._deviceVariableFunctionName(t, ["getProperty"])
                        if not py_func:
                            self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._directed_graph_out_iterator_var}' graph iterable object")
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
                    py_func = self._deviceVariableFunctionName(t, ["setVariable"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in '{self._output_message_var}' message output object")
                    # proceed
                    self.write(py_func)
                
            # standalone message input variable arg
            elif t.value.id in self._standalone_message_var:
                # check for legit FGPU function calls and translate
                self.write(f"{t.value.id}.")
                if t.attr in self.fgpu_input_msg_funcs: 
                    # proceed
                    self.write(t.attr)
                else:
                    # possible getter setter type function
                    py_func = self._deviceVariableFunctionName(t, ["getVariable"])
                    if not py_func:
                        self.RaiseError(t, f"Function '{t.attr}' does not exist in '{t.value.id}' message input object")
                    # proceed
                    self.write(py_func)
                    
            # standalone graph property arg
            elif t.value.id in self._directed_graph_vars:
                    self.write(f"{t.value.id}.")
                    # check for legit FGPU function calls and translate
                    if t.attr in self.fgpu_env_directed_graph_funcs:
                        # proceed
                        self.write(t.attr)
                    else:
                        # possible getter setter type function
                        py_func = self._deviceVariableFunctionName(t, ["getEdgeProperty"])
                        if not py_func:                        
                            py_func = self._deviceVariableFunctionName(t, ["getVertexProperty"])
                        if not py_func:
                            self.RaiseError(t, f"Function '{t.attr}' does not exist in '{t.value.id}' graph object")
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
            # allow any call on any locals (too many cases to enforce without type checking)
            elif t.value.id in self._locals:
                self.write(f"{t.value.id}.{t.attr}")
            else:
                self.RaiseError(t, f"Global '{t.value.id}' identifier not supported")
        # Call is a very nested situation which can occur only on macro environment properties. E.g. 'pyflamegpu.environment.getMacroPropertyInt('a').exchange(10)'
        elif isinstance(t.value, ast.Call):
            # handle the call by recursively calling this function to do the depth first execution of pyflamegpu.environment.getMacroPropertyInt('a')
            self.dispatchMemberFunction(t.value.func, t.value)
            # check that the handler was actually for macro environment 
            if t.value.call_type != "MacroEnvironment" :
                self.RaiseError(t, f"Function call {t.attr} is not supported")
            # now append the outer call by making sure the thing been called is a valid macro env function
            if not t.attr in self.fgpu_env_macro_funcs:
                self.RaiseError(t, f"Function {t.attr} is not a valid macro environment function")
            # write inner call args
            self.write("(")
            self._CallArguments(t.value)
            self.write(")")
            # write outer function (call args will be completed by _Call)
            self.write(f".{t.attr}")
            
   
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
        # Catch odd case of multi line strings and doc strings which are Expr with a Constant string type value
        if isinstance(tree.value, ast.Constant):
            if isinstance(tree.value.value, str):
                return
        # catch special case of Python 3.7 Where doc string is a Str and not a Constant
        elif sys.version_info < (3,8,0) and isinstance(tree.value, ast.Str): # num required for python 3.7
            return 
        # otherwise treat like a normal expression
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
            # Special case, catch message.at() where a message is returned outside a message loop
            if hasattr(t.value, "func") and isinstance(t.value.func, ast.Attribute):
                if t.value.func.attr == 'at' :
                    if t.value.func.value.id == self._input_message_var :
                        self._standalone_message_var.append(t.targets[0].id)
                # Special case, track which variables hold directed graph handles
                elif t.value.func.attr == 'getDirectedGraph' :
                    if t.value.func.value.value.id == "pyflamegpu" and t.value.func.value.attr == "environment" :
                        self._directed_graph_vars.append(t.targets[0].id)
            # Special case, definitions outside of agent fn are made const
            if self._indent == 0:
                self.write("constexpr ")
            self.write("auto ")
            self._locals[t.targets[0].id] =  self._indent
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
        Checks the decorators of the function definition much must be either 'pyflamegpu.agent_function', 'pyflamegpu.agent_function_condition' or 'pyflamegpu.device_function'.
        Each is then processed in a different way using a specific dispatcher.
        Function calls are actually checked and only permitted (or user defined) function calls are supported.
        """
        self.write("\n")
        # check decorators
        if len(t.decorator_list) != 1 or not isinstance(t.decorator_list[0], ast.Attribute):
            self.RaiseError(t, "Function definitions require a single pyflamegpu decorator of either 'pyflamegpu.agent_function', 'pyflamegpu.agent_function_condition' or 'pyflamegpu.device_function'")       
        # FLAMEGPU_AGENT_FUNCTION
        if t.decorator_list[0].attr == 'agent_function' and t.decorator_list[0].value.id == 'pyflamegpu':
            if getattr(t, "returns", False):
                self.RaiseWarning(t, "Function definition return type not supported on 'pyflamegpu.agent_function'")
            self.fill(f"FLAMEGPU_AGENT_FUNCTION({t.name}, ")
            self.dispatchFGPUFunctionArgs(t)
            self.write(")")
        # FLAMEGPU_DEVICE_FUNCTION
        elif t.decorator_list[0].attr == 'device_function' and t.decorator_list[0].value.id == 'pyflamegpu':
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
        # FLAMEGPU_DEVICE_FUNCTION
        elif t.decorator_list[0].attr == 'agent_function_condition' and t.decorator_list[0].value.id == 'pyflamegpu':
            # check for return annotation
            if not hasattr(t, "returns"):
                self.RaiseError(t, "Agent function conditions must have a 'bool' return type specified as a return type annotation")
            # check for return annotation type
            if not isinstance(t.returns, ast.Name):
                self.RaiseError(t, "Agent function conditions return type must be 'bool'")
            if t.returns.id != 'bool':
                self.RaiseError(t, "Agent function conditions return type must be 'bool'")
            # check to ensure no arguments (discard any with a warning)
            if t.args.args:
                self.RaiseWarning(t, "Agent function conditions does not support arguments. These will be discarded.")
            # write the agent function macro
            self.fill(f"FLAMEGPU_AGENT_FUNCTION_CONDITION({t.name})")
        else:
            self.RaiseError(t, "Function definition uses an unsupported decorator. Must use either 'pyflamegpu.agent_function', 'pyflamegpu.agent_function_condition' or 'pyflamegpu.device_function'")
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _AsyncFunctionDef(self, t):
        self.RaiseError(t, "Async functions not supported")

    def _For(self, t):
        """
        Two type for for loop are supported. Either;
        1) Message for loop in which case the format requires a iterator using the named pyflamegpu function argument of 'message_in'
        2) A range based for loop with 1 to 3 arguments which is converted into a c style loop
        """
        # if message or graph loop then process differently
        if isinstance(t.iter, ast.Name):
            if t.iter.id == self._input_message_var:
                self.dispatchMessageLoop(t)
            elif t.iter.id in self._directed_graph_vars:
                self.dispatchGraphLoop(t)
            else:
                self.RaiseError(t, "Range based for loops only support message iteration using 'message_in' or directed graph iterator")
        # do not support for else
        elif t.orelse:
            self.RaiseError(t, "For else not supported")
        # allow calls but only to range function
        elif isinstance(t.iter, ast.Call):
            # simple function call e.g. message_in() or range()
            if isinstance(t.iter.func, ast.Name):
                # catch case of message_input with arguments (e.g. spatial messaging)
                if t.iter.func.id == self._input_message_var:
                    self.dispatchMessageLoop(t)
                elif t.iter.func.id in self._directed_graph_vars:
                    self.dispatchGraphLoop(t)
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
            # member function call can only be on message_in.func() type call.
            elif isinstance(t.iter.func, ast.Attribute):
                # must be an attribute (e.g. calling a member of message_in)
                if t.iter.func.value.id == self._input_message_var:
                    self.dispatchMessageLoop(t)
                elif t.iter.func.value.id in self._directed_graph_vars:
                    self.dispatchGraphLoop(t)
                else:
                    self.RaiseError(t, "Range based for loops only support calling members of message input variable")
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
        self.RaiseError(t, "Byte strings and Bytes function not supported")

    def _Str(self, tree):
        # force writing in double quotes
        self.write(f'"{tree.s}"')
        
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
        # Required only for Python 3.7
        if t.value == None:
            self.write(0)
        elif t.value:
            self.write("true")
        else:
            self.write("false")

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
            self.RaiseError(t, "Byte strings and Bytes function not supported")
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
         * pyflamegpu.attribute - a supported attribute e.g. pyflamegpu.ALIVE. This will be translated into a namespace member.
         * math.constant - Any supported math constants are translated to C definition versions
        """
        # Only a limited set of globals supported
        func_dict = None
        
        # pyflamegpu singleton
        if isinstance(t.value, ast.Name):
            if t.value.id == "pyflamegpu":
                if t.attr in self.fgpu_attrs:
                    # proceed
                    self.write("flamegpu::")
                    self.write(t.attr)
                else:
                    self.RaiseError(t, f"Attribute '{t.attr}' does not exist in pyflamegpu object")
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

    def _CallArguments(self, t):
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
    
    def _Call(self, t):
        """
        Some basic checks are undertaken on calls to ensure that the function being called is either a builtin or defined device function.
        A special dispatcher is required 
        """
        # check calls but let attributes check in their own dispatcher
        funcs = self._device_functions + self.pythonbuiltins + [self._input_message_var] # message_input variable is a valid function name as certain message types have arguments on iterator
        if isinstance(t.func, ast.Name):
            if t.func.id in self.basic_arg_types:
                # Special case for casting python basic types (int/float)
                self.write(f"static_cast<{t.func.id}>")
            else:
                if (t.func.id not in funcs):
                    self.RaiseWarning(t, f"Function call to '{t.func.id}' is not a defined FLAME GPU device function or a supported python built in.")
                # dispatch even if warning raised
                self.dispatch(t.func)
        elif isinstance(t.func, ast.Lambda):
            self.dispatch(t.func) # not supported
        else:
            # special handler for dispatching member function calls
            # This would otherwise be an attribute
            self.dispatchMemberFunction(t.func, t)        
        self.write("(")
        self._CallArguments(t)
        self.write(")")

    def _Subscript(self, t):
        """
        Arrays are not supported but subscript allows accessing array like variables which is required for macro environment properties (e.g. a[0][1][2])
        Obvious limitation is no slicing type syntax (e.g. a[:2])
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
