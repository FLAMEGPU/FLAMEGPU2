/*
 * CUDAAgentModel.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef CUDAAGENTMODEL_H_
#define CUDAAGENTMODEL_H_

class CUDAAgentModel {
public:
	CUDAAgentModel();
	virtual ~CUDAAgentModel();

	void RegisterModelDescription();
};

#endif /* CUDAAGENTMODEL_H_ */
