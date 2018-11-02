#pragma once

/**
 * @file 
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */


#include <string>
#include "statereader.h"
#include "statewriter.h"
#include "xmlReader.h"
#include "xmlWriter.h"


using namespace std;

/**
* Abstract factory defines methods to create all
* related products.
*/
class Factory {
public:
	virtual StateReader *create_xml_reader(const ModelDescription &model, const char *input) = 0;
	virtual StateReader *create_hdf_reader(const ModelDescription &model, const char *input) = 0;

	virtual StateWriter *create_xml_writer() = 0;
	virtual StateWriter *create_hdf_writer() = 0;
};

/**
* Concrete factory creates concrete products, but
* returns them as abstract.
*/
class ReaderFactory : public Factory {
public:
	StateReader *create_xml_reader(const ModelDescription &model, const char *input) {
		return new xmlReader(model,input);
	}
	//StateReader *create_hdf(const ModelDescription &model, const char *input) {
		//return new hdfReader(model,input);}
};

/**
* Concrete factory creates concrete products, but
* returns them as abstract.
*/
class WriterFactory : public Factory {
public:
	StateWriter *create_xml_writer();
	StateWriter *create_hdf_writer();
};


/**
* Concrete factory creates concrete products, but
* returns them as abstract.
*/
class ReaderFactory1 {
public:
	StateReader *create_xml(const ModelDescription &model, const char *input) {
		return new xmlReader(model, input);
	}
	//StateReader *create_hdf(const ModelDescription &model, const char *input) {
	//return new hdfReader(model,input);}
};


class WriterFactory1 {
public:
	StateWriter *write_xml(const ModelDescription &model, const char *input) {
		return new xmlWriter(model, input);
	}
};
