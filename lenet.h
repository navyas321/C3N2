#ifndef LENET_H
#define LENET_H
#include "ap_fixed.h"
#include "ap_int.h"

typedef half hw_data_t;
typedef float sw_data_t;


hw_data_t relu(hw_data_t x);
hw_data_t relugrad(hw_data_t y);
void convolution1(hw_data_t input[1][32][32], hw_data_t layer1[6][28][28], hw_data_t weight0_1[1][6][5][5], hw_data_t bias0_1[6]);
void subsamp_max_forward1(hw_data_t layer1[6][28][28], hw_data_t layer2[6][14][14]);
void convolution3(hw_data_t layer2[6][14][14], hw_data_t layer3[16][10][10], hw_data_t weight2_3[6][16][5][5], hw_data_t bias2_3[16]);
void subsamp_max_forward2(hw_data_t layer3[16][10][10], hw_data_t layer4[16][5][5]);
void convolution5(hw_data_t layer4[16][5][5], hw_data_t layer5[120][1][1], hw_data_t weight4_5[16][120][5][5], hw_data_t bias4_5[120]);
void fc6(hw_data_t layer5[120][1][1], hw_data_t output[10], hw_data_t weight5_6[120][10], hw_data_t bias5_6[10]);
void fc6_backwards(hw_data_t layer5[120][1][1], hw_data_t layer5_err[120][1][1], hw_data_t output_err[10], hw_data_t weight5_6[120][10], hw_data_t weight5_6_delta[120][10], hw_data_t bias5_6_delta[10]);
void subsamp_max_backward1 (hw_data_t A[16][10][10], hw_data_t B[16][10][10], hw_data_t C[16][5][5]);
void subsamp_max_backward2 (hw_data_t A[6][28][28], hw_data_t B[6][28][28], hw_data_t layer2_err[6][14][14]);
void convolution_backward1 (hw_data_t A[16][5][5], hw_data_t B[16][5][5], hw_data_t layer5_err[120][1][1], hw_data_t weight4_5[16][120][5][5], hw_data_t weight4_5_delta[16][120][5][5], hw_data_t bias4_5_delta[120]);
void convolution_backward2 (hw_data_t layer2[6][14][14], hw_data_t layer2_err[6][14][14], hw_data_t layer3_err[16][10][10], hw_data_t weight2_3[6][16][5][5], hw_data_t weight2_3_delta[6][16][5][5], hw_data_t bias2_3_delta[16]);
void convolution_backward3 (hw_data_t input[1][32][32], hw_data_t input_err[1][32][32], hw_data_t layer1_err[6][28][28], hw_data_t weight0_1[1][6][5][5], hw_data_t weight0_1_delta[1][6][5][5], hw_data_t bias0_1_delta[6]);
void call(volatile sw_data_t lenet[51902], volatile sw_data_t deltas[51902], volatile sw_data_t errors[9035], volatile sw_data_t features[9035]);
#endif
