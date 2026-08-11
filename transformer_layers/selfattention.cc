#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
//#include <cstdint>
#include "debuggerFunctions.h"


#define W_DATA 4

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t **weightVector, std::size_t kernel_dim, std::size_t max_col) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    kernel_size_ = kernel_dim;
    max_col_ = max_col;

    // printf("Creating QUERY layer with:\n- input_dim = \t %ld\n- head_hidden_size = \t %ld\n", input_dim, head_hidden_size);
    query_layer = new Dense(input_dim, head_hidden_size, weightVector[0]);
    key_layer = new Dense(input_dim, head_hidden_size, weightVector[1]);
    value_layer = new Dense(input_dim, head_hidden_size, weightVector[2]);
    softmax = new Softmax();

    query_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    key_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    value_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    attention_scores = new uint32_t[pre_seq_len * pre_seq_len >> 2]();
}

SingleHeadSelfAttn::~SingleHeadSelfAttn() {

    delete[] query_layer_out;
    delete[] key_layer_out;
    delete[] key_transposed_layer_out;
    delete[] value_layer_out;
    delete[] attention_scores;

    delete query_layer;
    delete key_layer;
    delete value_layer;
    delete softmax;
}


#include  "../transformer.h"

void SingleHeadSelfAttn::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    
    // printf("outPtr = %p\n", (void*)query_layer_out);
    // printf("Query compute() with:\n- seq len = \t %ld\n", seq_len);
    query_layer->compute(seq_len, input, query_layer_out);
    printf("\n==========================\n");
    printf("QUERY LAYER OUT:\n");
    int8_t *res_p_q = (int8_t*)query_layer_out; 


    // for(int i=0; i<(D_SEQ * (D_Q / 4)); i++){
    //     printf("[%d]\t", i);
    //     std::cout << query_layer_out[i] << " --> ";
    //     int8_t *res_p_q = (int8_t*)&query_layer_out[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_p_q[j]);
    //     }
    //     printf("\n");
    // }
    // exit(0);

    printf("Key\n");
    key_layer->compute(seq_len, input, key_layer_out);
    int8_t *res_p_k = (int8_t*)key_layer_out; 
    printf("\n==========================\n");
    printf("KEY LAYER OUT:\n");


    // for(int i=0; i<(D_SEQ * (D_Q / 4)); i++){
    //     printf("[%d]\t", i);
    //     std::cout << key_layer_out[i] << " --> ";
    //     int8_t *res_p_q = (int8_t*)&key_layer_out[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_p_q[j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // exit(0);

    // printf("Value\n");
    // value_layer->compute(seq_len, input, value_layer_out);
    // printf("\n==========================\n");
    // printf("VALUE LAYER OUT:\n");

    // for(int i=0; i<(D_SEQ * (D_Q / 4)); i++){
    //     printf("[%d]\t", i);
    //     std::cout << value_layer_out[i] << " --> ";
    //     int8_t *res_p_q = (int8_t*)&value_layer_out[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_p_q[j]);
    //     }
    //     printf("\n");
    // }

    // exit(0);


#ifdef BWMA

    // printf("Key layer out:\n");
    // for(int i=0; i<(pre_seq_len_ * head_hidden_size_ >> 2); i++){
    //     printf("[%d]\t", i);
    //     std::cout << key_layer_out[i] << " --> ";
    //     int8_t *res_p_q = (int8_t*)&key_layer_out[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_p_q[j]);
    //     }
    //     printf("\n");
    // }

    std::cout << "BWMA method" << std::endl;

    #ifdef NANO_3D
    printf("3D nano\n");
    Transpose::transpose_rearranged_3Dnano(key_layer_out, key_transposed_layer_out, head_hidden_size_,
                                    pre_seq_len_, SA_H, SA_W, W_DATA, max_col_);
    #else
    Transpose::transpose_rearranged(key_layer_out, key_transposed_layer_out, head_hidden_size_,
                                    pre_seq_len_, kernel_size_, max_col_);
    #endif


    // printf("\nKey TRANSPOSED layer out:\n");
    // for(int i=0; i<(pre_seq_len_ * head_hidden_size_ >> 2); i++){
    //     printf("[%d]\t", i);
    //     std::cout << key_transposed_layer_out[i] << std::endl;
    // }


#ifdef SIMD
    simdComputeBWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
                          head_hidden_size_, seq_len);
#else

    #ifdef NANO_3D
        smmComputeBWMA_3DLayers(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                        seq_len);
    #else
        smmComputeBWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                    seq_len);
    #endif
    


    // printf("QK \n");
    // int8_t *res_qk = (int8_t*)attention_scores; 
    // printf("\n==========================\n");
    // printf("QK \n LAYER OUT:\n");


    // for(int i=0; i<(pre_seq_len_ * (pre_seq_len_ / 4)); i++){
    //     printf("[%d]\t", i);
    //     std::cout << attention_scores[i] << " --> ";
    //     int8_t *res_qk = (int8_t*)&attention_scores[i]; 
    //     for(int j=0; j<4; j++){
    //         printf("%d, ", res_qk[j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");


    exit(0);

#endif // SIMD


    softmax->computeRearranged(attention_scores, seq_len, kernel_size_);

#ifdef SIMD
    simdComputeBWMA(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
#else
    smmComputeBWMA(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
#endif // SIMD




#else // BWMA
    std::cout<< "RWMA method" << std::endl;
    Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_,
                                    pre_seq_len_);


                                    #ifdef SIMD
    simdComputeRWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
               head_hidden_size_, seq_len);
#else
    smmComputeRWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
                head_hidden_size_, seq_len);
#endif // SIMD


    softmax->compute(attention_scores, seq_len);


    
#ifdef SIMD
    simdComputeRWMA(seq_len, attention_scores, output, value_layer_out,
               seq_len, head_hidden_size_);
#else
    smmComputeRWMA(seq_len, attention_scores, output, value_layer_out,
                seq_len, head_hidden_size_);
#endif // SIMD


#endif // BWMA

    softmax->post_softmax(output, seq_len, head_hidden_size_);
}