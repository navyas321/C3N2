#include "lenet.h"


hw_data_t relu(hw_data_t x)
{
#pragma HLS INLINE
    return x*(x > 0);
}

hw_data_t relugrad(hw_data_t y)
{
#pragma HLS INLINE
    return y > 0;
}

hw_data_t weight0_1[1][6][5][5];
hw_data_t weight2_3[6][16][5][5];
hw_data_t weight4_5[16][120][5][5];
hw_data_t weight5_6[120][10];
hw_data_t bias0_1[6];
hw_data_t bias2_3[16];
hw_data_t bias4_5[120];
hw_data_t bias5_6[10];
hw_data_t input[1][32][32];
hw_data_t layer1[6][28][28];
hw_data_t layer2[6][14][14];
hw_data_t layer3[16][10][10];
hw_data_t layer4[16][5][5];
hw_data_t layer5[120][1][1];
hw_data_t output[10];
hw_data_t err_layer5[120][1][1];
hw_data_t err_output[10];
hw_data_t del_weight5_6[120][10];
hw_data_t del_bias5_6[10];
hw_data_t err_layer3[16][10][10];
hw_data_t err_layer4[16][5][5];
hw_data_t err_layer1[6][28][28];
hw_data_t err_layer2[6][14][14];
hw_data_t del_weight4_5[16][120][5][5];
hw_data_t del_bias4_5[120];
hw_data_t del_weight2_3[6][16][5][5];
hw_data_t del_bias2_3[16];
hw_data_t err_input[1][32][32];
hw_data_t del_weight0_1[1][6][5][5];
hw_data_t del_bias0_1[6];


void convolution1(hw_data_t input[1][32][32], hw_data_t layer1[6][28][28], hw_data_t weight0_1[1][6][5][5], hw_data_t bias0_1[6])
//void convolution1()
{
	for(int y = 0; y < 6; ++y)
	{
		for(int o0 = 0;  o0 < 28; ++o0)
		{
			for(int o1 = 0;  o1 < 28; ++o1)
			{
				hw_data_t sum = bias0_1[y];
				for(int w0 = 0;  w0 < 5; ++w0)
				{
#pragma HLS PIPELINE
					hw_data_t tempI[5];
					for(int w1 = 0;  w1 < 5; ++w1)
					{
#pragma HLS UNROLL
						tempI[w1] = input[0][o0+w0][o1+w1] * weight0_1[0][y][w0][w1];
					}
					sum += (tempI[0] + tempI[1] + tempI[2] + tempI[3] + tempI[4]);
				}
				layer1[y][o0][o1] = relu(sum);
			}
		}
	}
}

void subsamp_max_forward1(hw_data_t layer1[6][28][28], hw_data_t layer2[6][14][14])
//void subsamp_max_forward1()
{
    for (int i=0; i<6; ++i) {
        for (int o0=0; o0<14; ++o0) {
            for (int o1=0; o1<14; ++o1) {
                int x0=0, x1=0, ismax;
                for (int l0=0; l0<2; ++l0) {
                    for (int l1=0; l1<2; ++l1) {
                        ismax = layer1[i][o0*2 + l0][o1*2 + l1] > layer1[i][o0*2 + x0][o1*2 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                layer2[i][o0][o1] = layer1[i][o0*2 + x0][o1*2 + x1];
            }
        }
    }
}

void convolution3(hw_data_t layer2[6][14][14], hw_data_t layer3[16][10][10], hw_data_t weight2_3[6][16][5][5], hw_data_t bias2_3[16])
//void convolution3()
{
	for(int y = 0; y < 16; ++y)
    {
		for(int o0 = 0; o0 < 10; ++o0)
        {
			for(int o1 = 0; o1 < 10; ++o1)
            {

				hw_data_t sum = bias2_3[y];
            	for(int x = 0; x < 6; ++x)
                {
            		hw_data_t tempO[5];
                    for(int w0 = 0;  w0 < 5; ++w0)
                    {
#pragma HLS PIPELINE
                    	hw_data_t tempI[5];
                        for(int w1 = 0;  w1 < 5; ++w1)
                        {
#pragma HLS UNROLL
                        	tempI[w1] = layer2[x][o0+w0][o1+w1] * weight2_3[x][y][w0][w1];
                        }
                        tempO[w0] = (tempI[0] + tempI[1] + tempI[2] + tempI[3] + tempI[4]);
                    }
                    sum += (tempO[0] + tempO[1] + tempO[2] + tempO[3] + tempO[4]);
                }
            	layer3[y][o0][o1] = relu(sum);
            }
        }
    }
}

void subsamp_max_forward2(hw_data_t layer3[16][10][10], hw_data_t layer4[16][5][5])
//void subsamp_max_forward2()
{
    for (int i=0; i<16; ++i) {
        for (int o0=0; o0<5; ++o0) {
            for (int o1=0; o1<5; ++o1) {
                int x0=0, x1=0, ismax;
                for (int l0=0; l0<2; ++l0) {
                    for (int l1=0; l1<2; ++l1) {
                        ismax = layer3[i][o0*2 + l0][o1*2 + l1] > layer3[i][o0*2 + x0][o1*2 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                layer4[i][o0][o1] = layer3[i][o0*2 + x0][o1*2 + x1];
            }
        }
    }
}

void convolution5(hw_data_t layer4[16][5][5], hw_data_t layer5[120][1][1], hw_data_t weight4_5[16][120][5][5], hw_data_t bias4_5[120])
//void convolution5()
{
	for(int y = 0; y < 120; ++y)
    {
		hw_data_t sum = bias4_5[y];
		for(int x = 0; x < 16; ++x)
		{
			for(int w0 = 0; w0 < 5; ++w0)
			{
				for(int w1 = 0; w1 < 5; ++w1)
				{
#pragma HLS PIPELINE
					sum += layer4[x][w0][w1] * weight4_5[x][y][w0][w1];
				}
			}
		}
		layer5[y][0][0] = relu(sum);
    }
}

void fc6(hw_data_t layer5[120][1][1], hw_data_t output[10], hw_data_t weight5_6[120][10], hw_data_t bias5_6[10])
//void fc6()
{
	for(int y = 0; y < 10; ++y)
    {
		hw_data_t sum = bias5_6[y];
		for(int x = 0; x < 120; ++x)
        {
#pragma HLS PIPELINE
            sum += layer5[x][0][0] * weight5_6[x][y];
        }
		output[y] = relu(sum);
    }
}

void fc6_backwards(hw_data_t layer5[120][1][1], hw_data_t err_layer5[120][1][1], hw_data_t err_output[10], hw_data_t weight5_6[120][10], hw_data_t del_weight5_6[120][10], hw_data_t del_bias5_6[10])
//void fc6_backwards()
{
    int x, y, j;
    for (x=0; x<120; ++x) {
    	hw_data_t sum = 0;
        for (y=0; y<10; ++y) {
            sum += err_output[y] * weight5_6[x][y];
        }
        err_layer5[x][0][0] = sum * relugrad(layer5[x][0][0]);
    }
//    for (i=0; i<120; ++i) {
//        err_layer5[i][0][0] *= relugrad(layer5[i][0][0]);
//    }
    for (j = 0; j < 10; ++j) {
        del_bias5_6[j] = err_output[j];
    }
    for (x=0; x<120; ++x) {
        for (y=0; y<10; ++y) {
            del_weight5_6[x][y] = layer5[x][0][0] * err_output[y];
        }
    }
}

void subsamp_max_backward1 (hw_data_t layer3[16][10][10], hw_data_t err_layer3[16][10][10], hw_data_t err_layer4[16][5][5])
//void subsamp_max_backward1 ()
{
    for (int i=0; i<16; ++i) {
        for (int o0=0; o0<5; ++o0) {
            for (int o1=0; o1<5; ++o1) {
                int x0=0, x1=0, ismax;
                for (int l0=0; l0<2; ++l0) {
                    for (int l1=0; l1<2; ++l1) {
                        ismax = layer3[i][o0*2 + l0][o1*2 + l1] > layer3[i][o0*2 + x0][o1*2 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                err_layer3[i][o0*2 + x0][o1*2 + x1] = err_layer4[i][o0][o1];
            }
        }
    }
}
void subsamp_max_backward2 (hw_data_t layer1[6][28][28], hw_data_t err_layer1[6][28][28], hw_data_t err_layer2[6][14][14])
//void subsamp_max_backward2 ()
{
    for (int i=0; i<6; ++i) {
        for (int o0=0; o0<14; ++o0) {
            for (int o1=0; o1<14; ++o1) {
                int x0=0, x1=0, ismax;
                for (int l0=0; l0<2; ++l0) {
                    for (int l1=0; l1<2; ++l1) {
                        ismax = layer1[i][o0*2 + l0][o1*2 + l1] > layer1[i][o0*2 + x0][o1*2 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                err_layer1[i][o0*2 + x0][o1*2 + x1] = err_layer2[i][o0][o1];
            }
        }
    }
}

void convolution_backward1 (hw_data_t layer4[16][5][5], hw_data_t err_layer4[16][5][5], hw_data_t err_layer5[120][1][1], hw_data_t weight4_5[16][120][5][5], hw_data_t del_weight4_5[16][120][5][5], hw_data_t del_bias4_5[120])
//void convolution_backward1 ()
{
    int x, y, i, w0, w1, o0, o1;
    for (x=0; x<16; ++x) {
		for(w0=0; w0<5; ++w0) {
			for(w1=0; w1<5; ++w1) {
				hw_data_t sum = 0;
				for (y=0; y<120; ++y) {
#pragma HLS PIPELINE II=4
					sum += err_layer5[y][0][0] * weight4_5[x][y][w0][w1];
				}
				err_layer4[x][w0][w1] = sum * relugrad(layer4[x][w0][w1]);
			}
        }
    }

//    for (i=0; i<16; ++i) {
//        for (j=0; j<5; ++j) {
//            for (k=0; k<5; ++k) {
//                err_layer4[i][j][k] *= relugrad(layer4[i][j][k]);
//            }
//        }
//    }

    for (i=0; i<120; ++i) {
		del_bias4_5[i] = err_layer5[i][0][0];
    }

    for (x=0; x<16; ++x) {
        for (y=0; y<120; ++y) {
            for (o0=0; o0<5; ++o0) {
                for (o1=0; o1<5; ++o1) {
#pragma HLS PIPELINE II=4
					del_weight4_5[x][y][o0][o1] = layer4[x][o0][o1] * err_layer5[y][0][0];
                }
            }
        }
    }
}

void convolution_backward2 (hw_data_t layer2[6][14][14], hw_data_t err_layer2[6][14][14], hw_data_t err_layer3[16][10][10], hw_data_t weight2_3[6][16][5][5], hw_data_t del_weight2_3[6][16][5][5], hw_data_t del_bias2_3[16])
//void convolution_backward2 ()
{
    int x, y, i, j, k, i0, i1, w0, w1, o0, o1;
    for (x=0; x<6; ++x) {
        for (y=0; y<16; ++y) {
            for(i0=0; i0<10; ++i0) {
                for(i1=0; i1<10; ++i1) {
                    for(w0=0; w0<5; ++w0) {
                        for(w1=0; w1<5; ++w1) {
#pragma HLS PIPELINE II=8
                            err_layer2[x][i0 + w0][i1 + w1] += err_layer3[y][i0][i1] * weight2_3[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }

    for (i=0; i<6; ++i) {
        for (j=0; j<14; ++j) {
            for (k=0; k<14; ++k) {
                err_layer2[i][j][k] *= relugrad(layer2[i][j][k]);
            }
        }
    }

    for (i=0; i<16; ++i) {
        for (j=0; j<10; ++j) {
            for (k=0; k<10; ++k) {
                del_bias2_3[i] += err_layer3[i][j][k];
            }
        }
    }

    for (x=0; x<6; ++x) {
        for (y=0; y<16; ++y) {
            for (o0=0; o0<5; ++o0) {
                for (o1=0; o1<5; ++o1) {
                	hw_data_t sum = 0;
                    for (w0=0; w0<10; ++w0) {
                        for (w1=0; w1<10; ++w1) {
#pragma HLS PIPELINE II=8
                        	sum += layer2[x][o0 + w0][o1 + w1] * err_layer3[y][w0][w1];
                        }
                    }
                    del_weight2_3[x][y][o0][o1] = sum;
                }
            }
        }
    }
}

void convolution_backward3 (hw_data_t input[1][32][32], hw_data_t err_input[1][32][32], hw_data_t err_layer1[6][28][28], hw_data_t weight0_1[1][6][5][5], hw_data_t del_weight0_1[1][6][5][5], hw_data_t del_bias0_1[6])
//void convolution_backward3 ()
{
    int x, y, i, j, k, i0, i1, w0, w1, o0, o1;
    for (x=0; x<1; ++x) {
        for (y=0; y<6; ++y) {
            for(i0=0; i0<28; ++i0) {
                for(i1=0; i1<28; ++i1) {
                    for(w0=0; w0<5; ++w0) {
                        for(w1=0; w1<5; ++w1) {
#pragma HLS PIPELINE II=8
                            err_input[x][i0 + w0][i1 + w1] += err_layer1[y][i0][i1] * weight0_1[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }

    for (i=0; i<1; ++i) {
        for (j=0; j<32; ++j) {
            for (k=0; k<32; ++k) {
                err_input[i][j][k] *= relugrad(input[i][j][k]);
            }
        }
    }

    for (i=0; i<6; ++i) {
        for (j=0; j<28; ++j) {
            for (k=0; k<28; ++k) {
                del_bias0_1[i] += err_layer1[i][j][k];
            }
        }
    }

    for (x=0; x<1; ++x) {
        for (y=0; y<6; ++y) {
            for (o0=0; o0<5; ++o0) {
                for (o1=0; o1<5; ++o1) {
                    for (w0=0; w0<28; ++w0) {
                        for (w1=0; w1<28; ++w1) {
#pragma HLS PIPELINE II=8
                            del_weight0_1[x][y][o0][o1] += input[x][o0 + w0][o1 + w1] * err_layer1[y][w0][w1];
                        }
                    }
                }
            }
        }
    }
}

void forward()
{
    convolution1(input, layer1, weight0_1, bias0_1);
    subsamp_max_forward1(layer1, layer2);
    convolution3(layer2, layer3, weight2_3, bias2_3);
    subsamp_max_forward2(layer3, layer4);
    convolution5(layer4, layer5, weight4_5, bias4_5);
    fc6(layer5, output, weight5_6, bias5_6);
}

void backward()
{
    fc6_backwards(layer5, err_layer5, err_output, weight5_6, del_weight5_6, del_bias5_6);
    convolution_backward1 (layer4, err_layer4, err_layer5, weight4_5, del_weight4_5, del_bias4_5);
    subsamp_max_backward1 (layer3, err_layer3, err_layer4);
    convolution_backward2 (layer2, err_layer2, err_layer3, weight2_3, del_weight2_3, del_bias2_3);
    subsamp_max_backward2 (layer1, err_layer1, err_layer2);
    convolution_backward3 (input, err_input, err_layer1, weight0_1, del_weight0_1, del_bias0_1);
}

void call(volatile sw_data_t lenet[51902], volatile sw_data_t deltas[51902], volatile sw_data_t errors[9035], volatile sw_data_t features[9035])
{
#pragma HLS INTERFACE m_axi depth=51902 port=lenet offset=slave bundle=data_a
#pragma HLS INTERFACE m_axi depth=51902 port=deltas offset=slave bundle=data_b
#pragma HLS INTERFACE m_axi depth=9035 port=errors offset=slave bundle=data_c
#pragma HLS INTERFACE m_axi depth=9035 port=features offset=slave bundle=data_d
#pragma HLS INTERFACE s_axilite register port=return bundle=CTL


    int a, b, c, d;

    for (a=0; a<1; ++a) {
        for (b=0; b<6; ++b) {
            for (c=0; c<5; ++c) {
                for (d=0; d<5; ++d) {
                    weight0_1[a][b][c][d] = lenet[0 + b*25 + c*5 + d];
                }
            }
        }
    }
    for (a=0; a<6; ++a) {
        for (b=0; b<16; ++b) {
            for (c=0; c<5; ++c) {
                for (d=0; d<5; ++d) {
                    weight2_3[a][b][c][d] = lenet[150 + a*16*25 + b*25 + c*5 + d];
                }
            }
        }
    }
    for (a=0; a<16; ++a) {
        for (b=0; b<120; ++b) {
            for (c=0; c<5; ++c) {
                for (d=0; d<5; ++d) {
#pragma HLS PIPELINE
                    weight4_5[a][b][c][d] = lenet[2550 + a*120*25 + b*25 + c*5 + d];
                }
            }
        }
    }
    for (a=0; a<120; ++a) {
        for (b=0; b<10; ++b) {
            weight5_6[a][b] = lenet[50550 + a*10 + b];
        }
    }

    for (a=0; a<6; ++a) {
        bias0_1[a] = lenet[51750 + a];
    }
    for (a=0; a<16; ++a) {
        bias2_3[a] = lenet[51756 + a];
    }
    for (a=0; a<120; ++a) {
        bias4_5[a] = lenet[51772 + a];
    }
    for (a=0; a<10; ++a) {
        bias5_6[a] = lenet[51892 + a];
    }

    for (a=0; a<1; ++a) {
        for (b=0; b<32; ++b) {
            for (c=0; c<32; ++c) {
                input[a][b][c] = features[0 + b*32 + c];
            }
        }
    }

    if(features[9034] == 0)
    {
        forward();
    }

    else
    {
    	for (a=0; a<6; ++a) {
			for (b=0; b<28; ++b) {
				for (c=0; c<28; ++c) {
					layer1[a][b][c] = features[1024 + a*28*28 + b*28 + c];
				}
			}
		}
		for (a=0; a<6; ++a) {
			for (b=0; b<14; ++b) {
				for (c=0; c<14; ++c) {
					layer2[a][b][c] = features[5728 + a*14*14 + b*14 + c];
				}
			}
		}
		for (a=0; a<16; ++a) {
			for (b=0; b<10; ++b) {
				for (c=0; c<10; ++c) {
					layer3[a][b][c] = features[6904 + a*100 + b*10 + c];
				}
			}
		}
		for (a=0; a<16; ++a) {
			for (b=0; b<5; ++b) {
				for (c=0; c<5; ++c) {
					layer4[a][b][c] = features[8504 + a*25 + b*5 + c];
				}
			}
		}
		for (a=0; a<120; ++a) {
			for (b=0; b<1; ++b) {
				for (c=0; c<1; ++c) {
					layer5[a][b][c] = features[8904 + a];
				}
			}
		}

		for (a=0; a<10; ++a) {
			output[a] = features[9024 + a];
		}

        for (a=0; a<1; ++a) {
            for (b=0; b<6; ++b) {
                for (c=0; c<5; ++c) {
                    for (d=0; d<5; ++d) {
                        del_weight0_1[a][b][c][d] = deltas[0 + b*25 + c*5 + d];
                    }
                }
            }
        }
//        for (a=0; a<6; ++a) {
//            for (b=0; b<16; ++b) {
//                for (c=0; c<5; ++c) {
//                    for (d=0; d<5; ++d) {
//                        del_weight2_3[a][b][c][d] = deltas[150 + a*16*25 + b*25 + c*5 + d];
//                    }
//                }
//            }
//        }
//        for (a=0; a<16; ++a) {
//            for (b=0; b<120; ++b) {
//                for (c=0; c<5; ++c) {
//                    for (d=0; d<5; ++d) {
//#pragma HLS PIPELINE
//                        del_weight4_5[a][b][c][d] = deltas[2550 + a*120*25 + b*25 + c*5 + d];
//                    }
//                }
//            }
//        }
//        for (a=0; a<120; ++a) {
//            for (b=0; b<10; ++b) {
//                del_weight5_6[a][b] = deltas[50550 + a*10 + b];
//            }
//        }

        for (a=0; a<6; ++a) {
            del_bias0_1[a] = deltas[51750 + a];
        }
        for (a=0; a<16; ++a) {
            del_bias2_3[a] = deltas[51756 + a];
        }
//        for (a=0; a<120; ++a) {
//            del_bias4_5[a] = deltas[51772 + a];
//        }
//        for (a=0; a<10; ++a) {
//            del_bias5_6[a] = deltas[51892 + a];
//        }

        for (a=0; a<1; ++a) {
            for (b=0; b<32; ++b) {
                for (c=0; c<32; ++c) {
                    err_input[a][b][c] = errors[0 + b*32 + c];
                }
            }
        }
//        for (a=0; a<6; ++a) {
//            for (b=0; b<28; ++b) {
//                for (c=0; c<28; ++c) {
//                    err_layer1[a][b][c] = errors[1024 + a*28*28 + b*28 + c];
//                }
//            }
//        }
        for (a=0; a<6; ++a) {
            for (b=0; b<14; ++b) {
                for (c=0; c<14; ++c) {
                    err_layer2[a][b][c] = errors[5728 + a*14*14 + b*14 + c];
                }
            }
        }
//        for (a=0; a<16; ++a) {
//            for (b=0; b<10; ++b) {
//                for (c=0; c<10; ++c) {
//                    err_layer3[a][b][c] = errors[6904 + a*100 + b*10 + c];
//                }
//            }
//        }
//        for (a=0; a<16; ++a) {
//            for (b=0; b<5; ++b) {
//                for (c=0; c<5; ++c) {
//                    err_layer4[a][b][c] = errors[8504 + a*25 + b*5 + c];
//                }
//            }
//        }
//        for (a=0; a<120; ++a) {
//            for (b=0; b<1; ++b) {
//                for (c=0; c<1; ++c) {
//                    err_layer5[a][b][c] = errors[8904 + a];
//                }
//            }
//        }

        for (a=0; a<10; ++a) {
            err_output[a] = errors[9024 + a];
        }

        backward();

        for (a=0; a<1; ++a) {
            for (b=0; b<6; ++b) {
                for (c=0; c<5; ++c) {
                    for (d=0; d<5; ++d) {
                        deltas[0 + b*25 + c*5 + d] = del_weight0_1[a][b][c][d];
                    }
                }
            }
        }
        for (a=0; a<6; ++a) {
            for (b=0; b<16; ++b) {
                for (c=0; c<5; ++c) {
                    for (d=0; d<5; ++d) {
                        deltas[150 + a*16*25 + b*25 + c*5 + d] = del_weight2_3[a][b][c][d];
                    }
                }
            }
        }
        for (a=0; a<16; ++a) {
            for (b=0; b<120; ++b) {
                for (c=0; c<5; ++c) {
                    for (d=0; d<5; ++d) {
#pragma HLS PIPELINE
                        deltas[2550 + a*120*25 + b*25 + c*5 + d] = del_weight4_5[a][b][c][d];
                    }
                }
            }
        }
        for (a=0; a<120; ++a) {
            for (b=0; b<10; ++b) {
                deltas[50550 + a*10 + b] = del_weight5_6[a][b];
            }
        }

        for (a=0; a<6; ++a) {
            deltas[51750 + a] = del_bias0_1[a];
        }
        for (a=0; a<16; ++a) {
            deltas[51756 + a] = del_bias2_3[a];
        }
        for (a=0; a<120; ++a) {
            deltas[51772 + a] = del_bias4_5[a];
        }
        for (a=0; a<10; ++a) {
            deltas[51892 + a] = del_bias5_6[a];
        }

        for (a=0; a<1; ++a) {
            for (b=0; b<32; ++b) {
                for (c=0; c<32; ++c) {
                    errors[0 + b*32 + c] = err_input[a][b][c];
                }
            }
        }
        for (a=0; a<6; ++a) {
            for (b=0; b<28; ++b) {
                for (c=0; c<28; ++c) {
                    errors[1024 + a*28*28 + b*28 + c] = err_layer1[a][b][c];
                }
            }
        }
        for (a=0; a<6; ++a) {
            for (b=0; b<14; ++b) {
                for (c=0; c<14; ++c) {
                    errors[5728 + a*14*14 + b*14 + c] = err_layer2[a][b][c];
                }
            }
        }
        for (a=0; a<16; ++a) {
            for (b=0; b<10; ++b) {
                for (c=0; c<10; ++c) {
                    errors[6904 + a*100 + b*10 + c] = err_layer3[a][b][c];
                }
            }
        }
        for (a=0; a<16; ++a) {
            for (b=0; b<5; ++b) {
                for (c=0; c<5; ++c) {
                    errors[8504 + a*25 + b*5 + c] = err_layer4[a][b][c];
                }
            }
        }
        for (a=0; a<120; ++a) {
            for (b=0; b<1; ++b) {
                for (c=0; c<1; ++c) {
                    errors[8904 + a] = err_layer5[a][b][c];
                }
            }
        }

        for (a=0; a<10; ++a) {
            errors[9024 + a] = err_output[a];
        }

    }

    for (a=0; a<1; ++a) {
        for (b=0; b<6; ++b) {
            for (c=0; c<5; ++c) {
                for (d=0; d<5; ++d) {
                    lenet[0 + b*25 + c*5 + d] = weight0_1[a][b][c][d];
                }
            }
        }
    }
    for (a=0; a<6; ++a) {
        for (b=0; b<16; ++b) {
            for (c=0; c<5; ++c) {
                for (d=0; d<5; ++d) {
                    lenet[150 + a*16*25 + b*25 + c*5 + d] = weight2_3[a][b][c][d];
                }
            }
        }
    }
    for (a=0; a<16; ++a) {
        for (b=0; b<120; ++b) {
            for (c=0; c<5; ++c) {
                for (d=0; d<5; ++d) {
#pragma HLS PIPELINE
                    lenet[2550 + a*120*25 + b*25 + c*5 + d] = weight4_5[a][b][c][d];
                }
            }
        }
    }
    for (a=0; a<120; ++a) {
        for (b=0; b<10; ++b) {
            lenet[50550 + a*10 + b] = weight5_6[a][b];
        }
    }

    for (a=0; a<6; ++a) {
        lenet[51750 + a] = bias0_1[a];
    }
    for (a=0; a<16; ++a) {
        lenet[51756 + a] = bias2_3[a];
    }
    for (a=0; a<120; ++a) {
        lenet[51772 + a] = bias4_5[a];
    }
    for (a=0; a<10; ++a) {
        lenet[51892 + a] = bias5_6[a];
    }

    for (a=0; a<1; ++a) {
        for (b=0; b<32; ++b) {
            for (c=0; c<32; ++c) {
                features[0 + b*32 + c] = input[a][b][c];
            }
        }
    }
    for (a=0; a<6; ++a) {
        for (b=0; b<28; ++b) {
            for (c=0; c<28; ++c) {
                features[1024 + a*28*28 + b*28 + c] = layer1[a][b][c];
            }
        }
    }
    for (a=0; a<6; ++a) {
        for (b=0; b<14; ++b) {
            for (c=0; c<14; ++c) {
                features[5728 + a*14*14 + b*14 + c] = layer2[a][b][c];
            }
        }
    }
    for (a=0; a<16; ++a) {
        for (b=0; b<10; ++b) {
            for (c=0; c<10; ++c) {
                features[6904 + a*100 + b*10 + c] = layer3[a][b][c];
            }
        }
    }
    for (a=0; a<16; ++a) {
        for (b=0; b<5; ++b) {
            for (c=0; c<5; ++c) {
                features[8504 + a*25 + b*5 + c] = layer4[a][b][c];
            }
        }
    }
    for (a=0; a<120; ++a) {
        for (b=0; b<1; ++b) {
            for (c=0; c<1; ++c) {
                features[8904 + a] = layer5[a][b][c];
            }
        }
    }

    for (a=0; a<10; ++a) {
        features[9024 + a] = output[a];
    }
}

