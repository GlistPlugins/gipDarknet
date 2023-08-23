/*
 * gipDarknet.cpp
 *
 *  Created on: Mar 11, 2023
 *      Author: Noyan Culum
 */

#include "gipDarknet.h"


gipDarknet::gipDarknet() {
}

gipDarknet::~gipDarknet() {
}

void gipDarknet::initialize() {
    std::string datacfg = "coco.data";
    std::string cfgfile = "yolov7-tiny.cfg";
    std::string weightfile = "yolov7-tiny.weights";
    float thresh = 0.5f;
    float hier_thresh = 0.5f;
    initialize((char*)datacfg.c_str(), (char*)cfgfile.c_str(), (char*)weightfile.c_str(), thresh, hier_thresh);
}

void gipDarknet::initialize(std::string dataCfg, std::string cfgFile, std::string weightFile, float thresh, float hierThresh) {
	dataCfg = gGetFilesDir() + dataCfg;
	char* datacfg = (char*)dataCfg.c_str();
	cfgFile = gGetFilesDir() + cfgFile;
	char* cfgfile = (char*)cfgFile.c_str();
	weightFile = gGetFilesDir() + weightFile;
	char* weightfile = (char*)weightFile.c_str();

	this->thresh = thresh;
	hier_thresh = hierThresh;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    names = get_labels(name_list);

    alphabet = load_alphabet();
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
}

void gipDarknet::detectObjectsYolo(std::string fileName, std::string outFile) {
	fileName = gGetImagesDir() + fileName;
	char* filename = (char*)fileName.c_str();
	outFile = gGetImagesDir() + outFile;
	char* outfile = (char*)outFile.c_str();

    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
    	if(!input) return;
        strncpy(input, filename, 256);
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        gLogi("gipDarknet") << input << " : Predicted in " << (what_time_is_it_now()-time) << " seconds.";

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        gLogi("gipDarknet") << "outfile:" << outfile << ".jpg";
        if (filename) break;
    }
}

void gipDarknet::detectObjectsYolo(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
    	if(!input) return;
        strncpy(input, filename, 256);
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];

        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        gLogi("gipDarknet") << input << " : Predicted in " << (what_time_is_it_now()-time) << " seconds.";

        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        gLogi("gipDarknet") << "outfile:" << outfile << ".jpg";
        if (filename) break;
    }
}
