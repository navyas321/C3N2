#include "lenet.h"
//#include <stdlib.h>
//#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cmath>
//#include <memory.h>

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000


#define LENGTH_KERNEL	5

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 2

typedef unsigned char uint8;
typedef uint8 image[28][28];


typedef struct LeNet5
{
    float weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    float weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    float weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    float weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

    float bias0_1[LAYER1];
    float bias2_3[LAYER3];
    float bias4_5[LAYER5];
    float bias5_6[OUTPUT];

}LeNet5;

typedef struct Feature
{
    float input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    float layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    float layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    float layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    float layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    float layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    float output[OUTPUT];
    float forward;
}Feature;

static inline void load_input(Feature *features, image input)
{
    float (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
    const long sz = sizeof(image) / sizeof(**input);
    float mean = 0, std = 0;
    int j, k;
//    printf("%d, %d",sizeof(image) / sizeof(*input),sizeof(*input) / sizeof(**input) );
    for (j=0; j<28; ++j) {
        for (k=0; k<28; ++k) {
            mean += input[j][k];
            std += input[j][k] * input[j][k];
        }
    }
    mean /= sz;
//    printf("%f, %f\n", std / sz, mean*mean);
    std = sqrtf(std / sz - mean*mean);
    for (j=0; j<28; ++j) {
        for (k = 0; k < 28; ++k) {
            layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
        }
    }
}

static inline void softmax(float input[OUTPUT], float loss[OUTPUT], int label, int count)
{
    float inner = 0;
    int i;
    for (i = 0; i < count; ++i)
    {
        float res = 0;
        for (int j = 0; j < count; ++j)
        {
            res += expf(input[j] - input[i]);
        }
        loss[i] = 1. / res;
        inner -= loss[i] * loss[i];
    }
    inner += loss[label];
    for (i = 0; i < count; ++i)
    {
        loss[i] *= (i == label) - loss[i] - inner;
    }
}

static void load_target(Feature *features, Feature *errors, int label)
{
    float *output = (float *)features->output;
    float *error = (float *)errors->output;
    softmax(output, error, label, 10);
}

static uint8 get_result(Feature *features, uint8 count)
{
    float *output = (float *)features->output;
    const int outlen = 10;
    uint8 result = 0;
    float maxvalue = *output;
    for (uint8 i = 1; i < count; ++i)
    {
        if (output[i] > maxvalue)
        {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}

static float f64rand()
{
    static int randbit = 0;
    if (!randbit)
    {
        srand((unsigned)time(0));
        for (int i = RAND_MAX; i; i >>= 1, ++randbit);
    }
    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)rand() << i;
    lvalue |= (unsigned long long)rand() >> -i;
    double t = *(double *)&lvalue - 3;
    return (float) t;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
    int i = 0;
    float buffer[51902] = { 0 };
    for (i = 0; i < batchSize; ++i)
    {
        Feature features = { 0 };
        Feature errors = { 0 };
        LeNet5	deltas = { 0 };
        load_input(&features, inputs[i]);
        features.forward = 0;
        call((float*)lenet, (float*)&deltas, (float*)&errors, (float*)&features);
        load_target(&features, &errors, labels[i]);
        features.forward = 1;
        call((float*)lenet, (float*)&deltas, (float*)&errors, (float*)&features);
        {
            for(int j=0; j<51902; ++j)
                buffer[j] += ((float *)&deltas)[j];
        }
    }
    float k = ALPHA / batchSize;
    for(i=0; i<51902; ++i)
        ((float *)lenet)[i] += k * buffer[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    features.forward = 0;
    call((float*)lenet, nullptr, nullptr, (float*)&features);
    return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
    float *pos;
    int *ipos;
    for (pos = (float *)lenet->weight0_1; pos < (float *)lenet->bias0_1; *pos++ = f64rand());
    for (pos = (float *)lenet->weight0_1; pos < (float *)lenet->weight2_3; *pos++ = *pos * (float) sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    for (pos = (float *)lenet->weight2_3; pos < (float *)lenet->weight4_5; *pos++ = *pos * (float) sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    for (pos = (float *)lenet->weight4_5; pos < (float *)lenet->weight5_6; *pos++ = *pos * (float) sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    for (pos = (float *)lenet->weight5_6; pos < (float *)lenet->bias0_1; *pos++ = *pos * (float) sqrt(6.0 / (LAYER5 + OUTPUT)));
    for (ipos = (int *)lenet->bias0_1; ipos < (int *)(lenet + 1); *ipos++ = 0);
}

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen((char *)data_file, "rb");
    FILE *fp_label = fopen((char *)label_file, "rb");
    if (!fp_image||!fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data)*count, 1, fp_image);
    fread(label,count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
    for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
    {
//        std::cout << i * 100 / total_size << std::endl;
        TrainBatch(lenet, train_data + i, train_label + i, batch_size);
        if (i * 100 / total_size > percent) {
            percent = i * 100 / total_size;
//			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent);
            std::cout << "batchsize: " << batch_size << "\ttrain: " << 100*i / total_size << std::endl;
        }
    }
    std::cout << "leaving for loop" << std::endl;

}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
    int right = 0, percent = 0;
    std::cout << "start training" << std::endl;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        right += l == p;
        if (i * 100 / total_size > percent) {
            percent = i * 100 / total_size;
//			printf("test:%2d%%\n", percent);
            std::cout << "test: " << 100*i / total_size << std::endl;

        }
    }
    return right;
}

int save(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen((char *)filename, "wb");
    if (!fp) return 1;
    fwrite(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen((char *)filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}



void foo()
{
    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    {
//		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n");
        std::cout << "ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n" << std::endl;
        free(train_data);
        free(train_label);
//		system("pause");
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
//		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n");
        std::cout << "ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n" << std::endl;
        free(test_data);
        free(test_label);
//		system("pause");
    }


    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (load(lenet, (char*) LENET_FILE))
        Initial(lenet);
//	clock_t start = clock();
    int batches[] = { 300 };
    for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
        training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
    int right = testing(lenet, test_data, test_label, COUNT_TEST);
//	printf("%d/%d\n", right, COUNT_TEST);
    std::cout << right << "/" << COUNT_TEST << std::endl;
//	printf("Time:%u\n", (unsigned)(clock() - start));
    //save(lenet, LENET_FILE);
    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);
//	system("pause");
}

int main(int argc, char **argv)
{
    std::cout << "begin running ... " << std::endl;
    foo();
    return 0;
}
