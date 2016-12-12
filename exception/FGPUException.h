
 /** FGPUException.cpp
 *
 * Created on: 09 Dec 2016
 */

#include <string>
#include <iostream>
#include <exception>

using namespace std;

class UnknownError {};

class FGPUException//: public exception
{
public:
    FGPUException() {};
    ~FGPUException() {};

    virtual const char *what() const
    {
        return "Unknown Error Message";
    }
};

class InvalidInputFile: public FGPUException
{
public:
InvalidInputFile():FGPUException(){}
virtual const char *what() const
    {
        return "Invalid Input File";
    }
};

class InvalidHashList : public FGPUException
{
public:
InvalidHashList():FGPUException(){}
virtual const char *what() const
    {
        return "Hash list full. This should never happen";
    }
};

class InvalidVarType : public FGPUException
{
public:
InvalidVarType():FGPUException(){}
virtual const char *what() const
    {
      return "Bad variable type in agent instance set/get variable";
    }
};

class InvalidStateName : public FGPUException
{
public:
InvalidStateName():FGPUException(){}
virtual const char *what() const
    {
      return "Invalid agent state name";
    }
};

class InvalidMapEntry : public FGPUException
{
public:
InvalidMapEntry():FGPUException(){}
virtual const char *what() const
    {
       return "Missing entry in type sizes map. Something went bad.";
    }
};

class InvalidAgentVar : public FGPUException
{
public:
 InvalidAgentVar():FGPUException(){}
virtual const char *what() const
    {
       return "Invalid agent memory variable";
    }
};

class InvalidCudaAgent: public FGPUException
{
public:
 InvalidCudaAgent():FGPUException(){}
virtual const char *what() const
    {
       return "CUDA agent not found. This should not happen";
    }
};

class InvalidCudaAgentMapSize : public FGPUException
{
public:
 InvalidCudaAgentMapSize():FGPUException(){}
virtual const char *what() const
    {
        return "CUDA agent map size is zero";
    }
};

class InvalidCudaAgentDesc : public FGPUException
{
public:
 InvalidCudaAgentDesc():FGPUException(){}
virtual const char *what() const
    {
        return "CUDA Agent uses different agent description";
    }
};

class InvalidAgentFunc : public FGPUException
{
public:
 InvalidAgentFunc():FGPUException(){}
virtual const char *what() const
    {
        return "Unknown agent function";
    }
};

class InvalidFuncLayerIndx : public FGPUException
{
public:
 InvalidFuncLayerIndx():FGPUException(){}
virtual const char *what() const
    {
        return "Agent layer index out of bounds!";
    }
};
