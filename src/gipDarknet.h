/*
 * gipDarknet.h
 *
 *  Created on: Mar 11, 2023
 *      Author: Noyan Culum
 */

#ifndef SRC_GIPDARKNET_H_
#define SRC_GIPDARKNET_H_

#include "gBasePlugin.h"
#include "darknet.h"


class gipDarknet : public gBasePlugin{
public:
	gipDarknet();
	virtual ~gipDarknet();

	void initialize(std::string versionId = "7-tiny");
	void initialize(std::string dataCfg, std::string cfgFile, std::string weightFile, float thresh = 0.5f, float hierThresh = 0.5f);

	void detectObjectsYolo(gImage* src);
	void detectObjectsYolo(std::string filename, std::string outfile);
	void detectObjectsYolo(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile);

	image gImageToDNImage(gImage* src);

private:
	char *datacfg;
	char *cfgfile;
	char *weightfile;
	float thresh;
	float hier_thresh;

	char **names;
    image **alphabet;
    network *net;

};

#endif /* SRC_GIPDARKNET_H_ */
