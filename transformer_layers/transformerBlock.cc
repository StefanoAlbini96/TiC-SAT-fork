//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, uint32_t ** weightVector,
                                   std::size_t kernelDim, std::size_t maxCol) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;

    for (int n =0; n< num_heads; n++){
        selfatten[n] = new SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, weightVector+n*3,
                                              kernelDim, maxCol);
    }

    condense = new Dense(num_heads* head_hidden_size, input_dim, weightVector[num_heads * 3]);

    multihead_out = new uint32_t[pre_seq_len * num_heads * head_hidden_size >> 2]();
    condense_out = new uint32_t[pre_seq_len * input_dim >> 2]();
    intermediateFF = new uint32_t[pre_seq_len * ff_size >> 2]();

#ifndef BWMA
    multihead_out_reshape = new uint32_t[pre_seq_len * num_heads * head_hidden_size >> 2]();
#endif

    addNorm = new AddNormalize(pre_seq_len, input_dim, kernelDim, maxCol);
    feedForward0 = new Dense(input_dim, ff_size, weightVector[num_heads * 3+ 1]);
    feedForward1 = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2]);
}

TransformerBlock::~TransformerBlock() = default;


void TransformerBlock::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    system("m5 resetstats");
    for (int n=0; n<num_heads_; n++){
        std::cout << "\n ------- Computing Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, multihead_out + n * (seq_len * head_hidden_size_ >> 2));

        
        // printf("\n SELF ATT\n");
        // printf("%d x %d\n", seq_len, head_hidden_size_ >> 2);
        // printf("==============\n");
        // for(int i=0; i<(seq_len * (head_hidden_size_ / 4)); i++){
        //     printf("[%d]\t", i);
        //     std::cout << multihead_out[i] << " --> ";
        //     int8_t *res_qk = (int8_t*)&multihead_out[i]; 
        //     for(int j=0; j<4; j++){
        //         printf("%d, ", res_qk[j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        // exit(0);
    }

    // printf("\n SELF ATT\n");
    // printf("%d x (%d x %d)\n", num_heads_, seq_len, head_hidden_size_ >> 2);
    // printf("==============\n");
    // for(int i=0; i<(num_heads_ * seq_len * (head_hidden_size_ / 4)); i++){
    //     printf("[%d]\t", i);
    //     std::cout << multihead_out[i] << " --> ";
    //     int8_t *res_qk = (int8_t*)&multihead_out[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_qk[j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // exit(0);

#ifndef BWMA
    Transpose::multihead_transpose(multihead_out, multihead_out_reshape,
                                   seq_len, head_hidden_size_ >> 2, num_heads_);
    multihead_out = multihead_out_reshape;
#endif

    std::cout << "\nCondense"  << std::endl;
    condense->compute(seq_len, multihead_out, condense_out);


    // printf("\n CONDENSE\n");
    // printf("(%ld x %ld)\n", seq_len, input_dim_ >> 2);
    // printf("==============\n");
    // for(int i=0; i<(seq_len * input_dim_ >> 2); i++){
    //     printf("[%d]\t", i);
    //     std::cout << multihead_out[i] << " --> ";
    //     int8_t *res_qk = (int8_t*)&multihead_out[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_qk[j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // exit(0);

    std::cout << "Add Norm"  << std::endl;
#ifdef BWMA

    #ifdef NANO_3D
        addNorm->computeRearranged_3Dnano(input, condense_out, SA_H, SA_W);
    #else
        addNorm->computeRearranged(input, condense_out);
    #endif


    printf("\n CONDENSE\n");
    printf("(%ld x %ld)\n", seq_len, input_dim_ >> 2);
    printf("==============\n");
    for(int i=0; i<(seq_len * input_dim_ >> 2); i++){
        printf("[%d]\t", i);
        std::cout << condense_out[i] << " --> ";
        int8_t *res_qk = (int8_t*)&condense_out[i]; 
        for(int j=0; j<4; j++){
            printf("%d, ", res_qk[j]);
        }
        printf("\n");
    }
    printf("\n");
    exit(0);

#else
    addNorm->compute(input, condense_out);
#endif

    system("m5 dumpresetstats");

    std::cout << "Feed Forward 0"  << std::endl;
    feedForward0->compute(seq_len, condense_out, intermediateFF);

    std::cout << "Feed Forward 1"  << std::endl;
    feedForward1->compute(seq_len, intermediateFF, output);

    std::cout << "Add Norm"  << std::endl;

#ifdef BWMA

    #ifdef NANO_3D
        addNorm->computeRearranged_3Dnano(condense_out, output, SA_H, SA_W);
    #else
        addNorm->computeRearranged(condense_out, output);
    #endif



    // printf("\n ADD NORM FINAL\n");
    // printf("(%ld x %ld)\n", seq_len, input_dim_ >> 2);
    // printf("==============\n");
    // for(int i=0; i<(seq_len * input_dim_ >> 2); i++){
    //     printf("[%d]\t", i);
    //     std::cout << multihead_out[i] << " --> ";
    //     int8_t *res_qk = (int8_t*)&multihead_out[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_qk[j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // exit(0);



#else
    addNorm->compute(condense_out, output);
#endif
    system("m5 dumpresetstats");

}
