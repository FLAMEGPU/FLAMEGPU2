#ifndef STATEREADER_H_
#define STATEREADER_H_

/**
 * @file statereader.h
 * @author  
 * @date    
 * @brief 
 *
 * \todo longer description
 */


#include "../xmlparser/tinyxml2.h"              //downloaded from https://github.com/leethomason/tinyxml2, the list of xml parsers : http://lars.ruoff.free.fr/xmlcpp/
#include "../gpu/CUDAErrorChecking.h"			//required for CUDA error handling functions
#include "cuRVE/curve.h"
#include "iterator"
#include "../exception/FGPUException.h"
#include "../model/AgentDescription.h"
#include "../pop/AgentPopulation.h"

using namespace std;
using namespace tinyxml2;

//TODO: Some example code of the handle class and an example function

class StateReader;  // Forward declaration (class defined below)


class StateReader 
{

public:

	// -----------------------------------------------------------------------
	//  Constructors and Destructor
	// -----------------------------------------------------------------------

    StateReader() {};
	//__device__ StateReader(unsigned int start= 0, unsigned int end = 0) : start_(start), messageList_size(end) {};

	~StateReader() {};

	// -----------------------------------------------------------------------
	//  The interface
	// -----------------------------------------------------------------------

	/*!
	 *
	 */
	virtual void parse(const char &source) = 0;
	

private:

	unsigned int start_;
	unsigned int end_;

	/* The copy constructor, you cannot call this directly */
	StateReader(const StateReader&);

	/* The assignment operator, you cannot call this directly */
	StateReader& operator=(const StateReader&);
};

/**
* \brief 
* \param source name of the inputfile
*/
void StateReader::parse(const inputSource &source)
{
	
}

void readInitialStates(const AgentDescription &agent_description, char* inputpath)
{
	/* Pointer to file */
	FILE *file;
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;

	int temp = 0;
	//int* itno = &temp;


	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Open config file to read-only */
	if ((file = fopen(inputpath, "r")) == NULL)
	{
		printf("Error opening initial states\n");
		exit(0);
	}

	reading = 1;

	// we need to know the variables 
	// check tinyxml test

	//using Tinyxml Library:
	XMLDocument doc;
	doc.LoadFile(inputpath);

	// Structure of the XML file:
	// -Element "states"        the root Element, which is the
	//                          FirstChildElement of the Document
	// --Element "itno"         child of the root states Element
	// --Element "environment"  child of the root states Element
	// --Element "xagent"       child of the root states Element
	// --- Text                 child of the xagent Element

	// Navigate to the itno, using the convenience function,
	const char* itno = doc.FirstChildElement("states")->FirstChildElement("itno")->GetText();
	printf("Agent number: %s\n", itno);

	// Text is just another Node to TinyXML-2. The more
	// general way to get to the XMLText:
	XMLText* textNode = doc.FirstChildElement("states")->FirstChildElement("TITLE")->FirstChild()->ToText();
	XMLText* textNode = doc.FirstChildElement("PLAY")->FirstChildElement("TITLE")->FirstChild()->ToText();
	itno = textNode->Value();
	printf("Name of play (2): %s\n", itno);




	/* Read file until end of xml */
	i = 0;
	while (reading == 1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);

		/* If the end of a tag */
		if (c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if (strcmp(buffer, "states") == 0) reading = 1;
			if (strcmp(buffer, "/states") == 0) reading = 0;
			if (strcmp(buffer, "itno") == 0) in_itno = 1;
			if (strcmp(buffer, "/itno") == 0) in_itno = 0;
			if (strcmp(buffer, "name") == 0) in_name = 1;
			if (strcmp(buffer, "/name") == 0) in_name = 0;
			if (strcmp(buffer, "/xagent") == 0)
			{
				if (strcmp(agentname, "Boid") == 0)
				{
					if (*h_xmachine_memory_Boid_count > xmachine_memory_Boid_MAX) {
						printf("ERROR: MAX Buffer size (%i) for agent Boid exceeded whilst reading data\n", xmachine_memory_Boid_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}

					h_Boids->id[*h_xmachine_memory_Boid_count] = Boid_id;
					h_Boids->x[*h_xmachine_memory_Boid_count] = Boid_x;//Check maximum x value
					if (agent_maximum.x < Boid_x)
						agent_maximum.x = (float)Boid_x;
					//Check minimum x value
					if (agent_minimum.x > Boid_x)
						agent_minimum.x = (float)Boid_x;

					h_Boids->y[*h_xmachine_memory_Boid_count] = Boid_y;//Check maximum y value
					if (agent_maximum.y < Boid_y)
						agent_maximum.y = (float)Boid_y;
					//Check minimum y value
					if (agent_minimum.y > Boid_y)
						agent_minimum.y = (float)Boid_y;

					h_Boids->z[*h_xmachine_memory_Boid_count] = Boid_z;//Check maximum z value
					if (agent_maximum.z < Boid_z)
						agent_maximum.z = (float)Boid_z;
					//Check minimum z value
					if (agent_minimum.z > Boid_z)
						agent_minimum.z = (float)Boid_z;

					h_Boids->fx[*h_xmachine_memory_Boid_count] = Boid_fx;
					h_Boids->fy[*h_xmachine_memory_Boid_count] = Boid_fy;
					h_Boids->fz[*h_xmachine_memory_Boid_count] = Boid_fz;
					(*h_xmachine_memory_Boid_count)++;
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
				Boid_id = 0;
				Boid_x = 0;
				Boid_y = 0;
				Boid_z = 0;
				Boid_fx = 0;
				Boid_fy = 0;
				Boid_fz = 0;

			}
			if (strcmp(buffer, "id") == 0) in_Boid_id = 1;
			if (strcmp(buffer, "/id") == 0) in_Boid_id = 0;
			if (strcmp(buffer, "x") == 0) in_Boid_x = 1;
			if (strcmp(buffer, "/x") == 0) in_Boid_x = 0;
			if (strcmp(buffer, "y") == 0) in_Boid_y = 1;
			if (strcmp(buffer, "/y") == 0) in_Boid_y = 0;
			if (strcmp(buffer, "z") == 0) in_Boid_z = 1;
			if (strcmp(buffer, "/z") == 0) in_Boid_z = 0;
			if (strcmp(buffer, "fx") == 0) in_Boid_fx = 1;
			if (strcmp(buffer, "/fx") == 0) in_Boid_fx = 0;
			if (strcmp(buffer, "fy") == 0) in_Boid_fy = 1;
			if (strcmp(buffer, "/fy") == 0) in_Boid_fy = 0;
			if (strcmp(buffer, "fz") == 0) in_Boid_fz = 1;
			if (strcmp(buffer, "/fz") == 0) in_Boid_fz = 0;


			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if (c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if (in_itno) *itno = atoi(buffer);
			if (in_name) strcpy(agentname, buffer);
			else
			{
				if (in_Boid_id) {
					Boid_id = (int)atoi(buffer);
				}
				if (in_Boid_x) {
					Boid_x = (float)atof(buffer);
				}
				if (in_Boid_y) {
					Boid_y = (float)atof(buffer);
				}
				if (in_Boid_z) {
					Boid_z = (float)atof(buffer);
				}
				if (in_Boid_fx) {
					Boid_fx = (float)atof(buffer);
				}
				if (in_Boid_fy) {
					Boid_fy = (float)atof(buffer);
				}
				if (in_Boid_fz) {
					Boid_fz = (float)atof(buffer);
				}

			}

			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if (in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);


	int size = 0;  // for now, to be replaced later

	/////////////////////
	AgentPopulation population1(agent_description, size);
	for (int i = 0; i< size; i++)
	{
		AgentInstance instance = population1.getNextInstance("default");
		instance.setVariable<float>("x", i*0.1f);
		instance.setVariable<float>("y", i*0.1f);
	}
	////////////////////////////////
}



#endif /* STATEREADER_H_ */
