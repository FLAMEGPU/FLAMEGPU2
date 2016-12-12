/*
 * FGPUException.cpp
 *
 *  Created on: 09 Dec 2016
 *      Author:
 */

#include <string>
#include <iostream>

using namespace std;


///////////////////////////////////////////// 1st solution
/*
class Overflow
{
public:
    Overflow() {};
    ~Overflow() {};
    const char *what() const
    {
        return "Overflow msg";
    }
};

class Underflow
{
public:
    Underflow() {};
    ~Underflow() {};
    const char *what() const
    {
        return "Underflow msg";
    }
};
*/

////////////////////////////// 2nd solution

class UnknownError {};

class FGPUException
{
public:
    FGPUException() {};
    ~FGPUException() {};

    virtual const char *what() const
    {
        std::cout << "Unknown Error Message\n";
        //return "Error msg";
    }

//protected:
};

class InvalidInputFile: public FGPUException
{
public:
InvalidInputFile():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Invalid Input File\n";
    }
};

class InvalidHashList : public FGPUException
{
public:
InvalidHashList():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Hash list full. This should never happen\n";
    }
};

class InvalidVarType : public FGPUException
{
public:
InvalidVarType():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Bad variable type in agent instance set variable\n";
    }
};

class InvalidStateName : public FGPUException
{
public:
InvalidStateName():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Invalid agent state name\n";
    }
};

class InvalidMapEntry : public FGPUException
{
public:
InvalidMapEntry():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Missing entry in type sizes map. Something went bad.\n";
    }
};

class InvalidAgentVar : public FGPUException
{
public:
 InvalidAgentVar():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Invalid agent memory variable\n";
    }
};

class InvalidCudaAgent: public FGPUException
{
public:
 InvalidCudaAgent():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "CUDA agent not found. This should not happen\n";
    }
};

class InvalidCudaAgentMapSize : public FGPUException
{
public:
 InvalidCudaAgentMapSize():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "CUDA agent map size is zero\n";
    }
};

class InvalidCudaAgentDesc : public FGPUException
{
public:
 InvalidCudaAgentDesc():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "CUDA Agent uses different agent description\n";
    }
};

class InvalidAgentFunc : public FGPUException
{
public:
 InvalidAgentFunc():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Unknown agent function\n";
    }
};

class InvalidFuncLayerIndx : public FGPUException
{
public:
 InvalidFuncLayerIndx():FGPUException(){}
virtual const char *what() const
    {
        std::cout << "Agent layer index out of bounds!\n";
    }
};





// USAGE for solution 2
/*
void MyFunc()
{
    int a = 1;
    cout<< "In MyFunc(). Throwing Overflow exception." << endl;
    throw Overflow();
}


 int main()
 { // we can throw class name in any func,, then catch it below
     try
     {
MyFunc();

     }
     catch (UnknownError)
     {
         std::cout << "Unable to process the error!\n";
     }
     catch (FGPUException& theException)
     {
         theException.what();
     }
     catch (...)
     {
         std::cout << "Something went wrong,"
             << "but I've no idea what!" << std::endl;
     }
     std::cout << "Done.\n";
     return 0;
 }
*/
