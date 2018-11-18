//
// Created by gzen on 31/10/18.
//
#include "load_param.h"

load_param::load_param(){
    load_path();
}

void load_param::load_path(){

    //set paths
    deposit = "C:/Users/morom/Desktop/";
	path_gen = deposit + "dataset/";
	path_orig = path_gen + "stampa_estratto/";
	path_affine = path_gen + "affine_matrix/";
	pairs = path_gen + "stampa_estratto/pairs.csv";
	numeric_results = deposit + "results/numeric_results";
	img_results = deposit + "results/img_results";

    //deposit = "/home/gzen/WORK/projects/CLEVER/PME1.0/";
    //path_gen = deposit + "Archive/";
    //path_orig = path_gen + "stampa_estratto_stable1/";
    //path_affine = path_gen + "stampa_risultati/affine_matrix/";
    //pairs = path_gen + "stampa_estratto_stable1/pairs.csv";



}
