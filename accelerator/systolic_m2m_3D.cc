/*
 * Copyright (c) 2026 EPFL
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Stefano Albini
 */
// Constructor.
#include <cstdint>
#include <iostream>
#include "systolic_m2m_3D.hh"



#ifdef NANO_3D


void print_buf_3Dnano(int8_t *ptr, int height, int width){
    int c = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d,  \t", ptr[c]);
            c++;
        
        }
        printf("\n");
    }
}




bool SystolicMatrixMultiplication_3D::loadWeights(int idx, uint32_t val, int layer) {
    
    printf("LOADING idx %d  |  layer = %d\n", idx, layer);
    for (int i=0; i < W_DATA; i++){
        auto currVal = (int8_t)((val >> (8 * (W_DATA -i-1))) & 0xff);
        // printf("%d\n", currVal);
        weights[layer][idx + i] = currVal;
    }

    // print_buf_3Dnano((int8_t*)weights[0], SA_H, SA_W);
    // printf("\n");

    if (val!=0)
        non_zero_tile = true;
    return non_zero_tile;
}


// Puts the new word of input and also reads a result word
uint32_t SystolicMatrixMultiplication_3D::inputQueue_3Dnano(int col, uint32_t val, int out_cnt) {


    printf("\nEnqueueing in col = %d  | out_Cnt = %d\n", col, out_cnt);

    // printf("Enqueueing in col = %d\n", col);
    // print_buf(inWaitingMemory, SA_H, SA_H);
    // printf("\n");

    for (int layer = 0; layer < N_3D_LAYERS; layer++){
        

        // Split the input to an array
        // printf("New val --> ");
        for (int i=0; i < W_DATA; i++){
            auto currVal = (int8_t)((val >> (8 * (W_DATA - i -1))) & 0xff);
            // printf("%d, ", currVal);
            int row_index = (col*W_DATA+i);
            mem2d(inWaitingMemory[layer], SA_H, row_index, SA_H - row_index - 1) = currVal; // off-diagonal of the waiting memory
        }
        // printf("\n");

        // print_buf(inWaitingMemory, SA_H, SA_H);   
    }


    // printf("reading result for col = %d and cnt = %d\n", col, output_cnt);

    int col_cnt = out_cnt % MAX_COL_OUT;
    int layer_cnt = out_cnt / MAX_COL_OUT;
    // printf("Reading out_cnt = %d  | col_cnt = %d  | layer_cnt = %d\n", out_cnt, col_cnt, layer_cnt);

    // Return the output
    uint32_t result = 0;
    // int result_idx = ((col+1)%MAX_COL) * W_DATA;
    int result_idx = ((col_cnt+1)%MAX_COL_OUT) * W_DATA;
    
    // int row_start = (1 - col_cnt) * W_DATA;
    int row_start = (MAX_COL_OUT - 1 - col_cnt) * W_DATA;
    int col_start = SA_W - 1 - row_start;


    // printf("Row start = (1 - %d) * 4 = %d\n", col_cnt, row_start);
    // printf("Col start =  %d - 1 - %d = %d\n", SA_W, row_start, col_start);

    // printf("LAY 0: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[0], SA_W, SA_W);
    // printf("\nLAY 1: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[1], SA_W, SA_W);


    // printf("Ciao\n");
    // printf("result idx = ((%d + 1) %d) * %d = %d\n", col, MAX_COL_OUT, W_DATA, ((col+1)%MAX_COL_OUT) * W_DATA);
    for (int i = 0; i < W_DATA; i++) {


        // int row = row_start + i;
        // int col = col_start - i;

        int row = row_start + (W_DATA-1-i);;
        int col = col_start - (W_DATA-1-i);

        // printf("row = %d\n", row);
        // printf("col = %d\n", col);
        result |=  mem2d(outWaitingMemory[layer_cnt], SA_W, row, col ) << (8 * (W_DATA - i - 1));
    }

    // printf("RESULT layer [%d], col [%d]\n", layer_cnt, col_start); 
    // int8_t *res = (int8_t*)&result; 
    // for(int j=0; j<W_DATA; j++){
    //     printf("%d, ", res[j]);
    // }
    // printf("\n");
    // printf("%u\n", result);

    return result;
}


void SystolicMatrixMultiplication_3D::printWeights() {
    std::cout << std::hex << (uint32_t) weights[0][0] << std::endl;
}




uint32_t SystolicMatrixMultiplication_3D::streamInOut_3Dnano(uint32_t val, int out_cnt) {

    printf("\n INPUTTING val = %u  | out_Cnt = %d\n", val, out_cnt);

    for (int layer = 0; layer < N_3D_LAYERS; layer++){
        // printf("------------------------\n FUNC\n");

        non_zero_tile = false;
        int col = MAX_COL - 1;
        // Split the input to an array
        // printf("New val --> ");
        for (int i=0; i < W_DATA; i++){
            auto currVal = (int8_t)((val >> (8 * (W_DATA - i -1))) & 0xff);
            // printf("%d, ", currVal);
            int row_index = (col*W_DATA+i);
            mem2d(inWaitingMemory[layer], SA_H, row_index, SA_H - row_index - 1) = currVal; // off-diagonal of the waiting memory
        }
        // printf("\n");

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, SA_H, SA_H);

        // printf("\nINPUT\n");
        // print_buf(inputMemory, SA_H, SA_W);

        // Shift the waiting memory to the right for skewing
        for (int i = 0; i < SA_H; i++) {
            mem2d(inputMemory[layer], SA_W, i, 0) = mem2d(inWaitingMemory[layer], SA_H, i, SA_H - 1);
            for (int j= SA_H - 1; j > 0; j--){ // TODO: shift only the right-hand triangle
                mem2d(inWaitingMemory[layer], SA_H, i, j) = mem2d(inWaitingMemory[layer], SA_H, i, j - 1);
            }
        }

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, SA_H, SA_H);

        // printf("\nINPUT\n");
        // print_buf(inputMemory, SA_H, SA_W);



        // printf("\nMult\n");
        int cnt = 0;

        // printf("\n---------------------------------------------------------\n");
        // Multiply the input to the weight and accumulate to the output
        for (int i= SA_H * SA_W - 1; i >= 0 ; i--){

            // printf("(%d * %d) + %d  \t|  ", inputMemory[layer][i], weights[layer][i], outputMemory[layer][i]);
            // if(cnt == (SA_W-1)){
            //     printf("\n");
            //     cnt = 0;
            // } else {
            //     cnt ++;
            // }

            outputMemory[layer][i + SA_W] = int(inputMemory[layer][i] * weights[layer][i]) + outputMemory[layer][i];
        }

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, SA_H, SA_H);

        // Shift the input memory to the right
        for (int i = 0; i < SA_H; i++) {
            for (int j= SA_W - 1; j > 0; j--){
                inputMemory[layer][i * SA_W + j] = inputMemory[layer][i * SA_W + j - 1];
            }
        }


        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, SA_H, SA_H);

        // printf("\nINPUT\n");
        // print_buf(inputMemory, SA_H, SA_W);


        // Shift the outWaitingMemory because of the skew in the output
        for (int j = 0; j < SA_W; j++) {
            for (int i= SA_W - 1; i > 0; i--){ // TODO: shift only the right-hand triangle
                mem2d(outWaitingMemory[layer], SA_W, i, j) = mem2d(outWaitingMemory[layer], SA_W, i - 1, j);
            }
        //    std::cout << std::hex << mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j) << std::endl;
            mem2d(outWaitingMemory[layer], SA_W, 0, j) = (uint8_t)(mem2d(outputMemory[layer], SA_W, SA_H, j) & 0xFF);
        }


        // printf("\n----- \nOUT WAIT\n");
        // print_buf((int8_t*)outWaitingMemory, SA_W, SA_W);



        // printf("\n----- \nOUT WAIT\n");
        // print_buf(outputMemory, SA_W, SA_H + 1);

        // printf("\n----- \nWAIT\n");
        // print_buf(inWaitingMemory, SA_H, SA_H);

    }

    // printf("\n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory, SA_W, SA_W);
    // printf("\n");

    // printf("reading result for cnt = %d\n", output_cnt);
    int col_cnt = out_cnt % MAX_COL_OUT;
    int layer_cnt = out_cnt / MAX_COL_OUT;
    // printf("Reading out_cnt = %d  | col_cnt = %d  | layer_cnt = %d\n", out_cnt, col_cnt, layer_cnt);


    // printf("LAY 0: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[0], SA_W, SA_W);
    // printf("\nLAY 1: \n");
    // print_buf_3Dnano((int8_t*)outWaitingMemory[1], SA_W, SA_W);

    // Return the output
    uint32_t result = 0;
    for (int i = 0; i < W_DATA; i++) {
        // printf("--> %d\n", mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ));
        result |=  mem2d(outWaitingMemory[layer_cnt], SA_W, SA_W - i - 1, i ) << (8 * (W_DATA - i - 1));
//        std::cout << std::hex << (int) mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ) << ",";
    }

    // printf("RESULT layer [%d]\n", layer_cnt);
    // int8_t *res = (int8_t*)&result; 
    // for(int j=0; j<W_DATA; j++){
    //     printf("%d, ", res[j]);
    // }
    // printf("\n");
    // printf("%u\n", result);

    
    return result;

}



#endif


