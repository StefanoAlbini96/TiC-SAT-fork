#include "transpose.h"
#include <iostream>

void Transpose::transpose(const uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) {
    uint32_t swap[4];
    std::size_t height_tile = height >> 2;
    std::size_t width_tile = width >> 2;
    for (int i=0; i < height_tile; i++){
        for (int j=0; j < width_tile; j++){
            for( int k=0; k< 4; k++){
                swap[k] = input[(i * 4 + k) * width_tile + j];
            }

            for ( int k=0; k< 4; k++){
                uint32_t result = 0;
                for (int s=0; s< 4; s++)
                    result |= ((swap [s] >> (24 - 8*k)) & 0xFF) << (24 - 8*s);
                output[(j * 4 + k) * height_tile + i] = result;
            }
        }
    }
}

void Transpose::transpose_rearranged(uint32_t* input, uint32_t* output, std::size_t width, std::size_t height,
                                     std::size_t kernelSize, std::size_t maxCol) {
    std::size_t tileRow = height / kernelSize;
    std::size_t tileCol = width / kernelSize;
    // printf("tileRow = %ld / %ld = %ld\n", height, kernelSize, tileRow);
    // printf("tileCol = %ld / %ld = %ld\n", width, kernelSize, tileCol);
    for (int i=0; i < tileRow; i++){
        for (int j=0; j < tileCol; j++){
            uint32_t * tileInputPtr = input + (j*tileRow + i) * (kernelSize * maxCol);
            uint32_t * tileOutputPtr = output + (i* tileCol + j) * (kernelSize * maxCol);
            for (int m=0; m< maxCol; m++){
                for (int indexIn4 =0; indexIn4<4; indexIn4++){
                    for ( int k=0; k< maxCol; k++){
                        uint32_t result = 0;
                        for (int s=0; s< 4; s++)
                            result |= ((*(tileInputPtr+(4*k+s) *maxCol + m)>> (24 - 8*indexIn4)) & 0xFF) << (24 - 8*s);
                        *(tileOutputPtr + m*4*maxCol + indexIn4* maxCol + k)= result;
                    }
                }
            }
        }
    }
}




void Transpose::transpose_rearranged_3Dnano(uint32_t* input, uint32_t* output, std::size_t width, std::size_t height,
                                     std::size_t kernelSize_h, std::size_t kernelSize_w, std::size_t w_data, std::size_t maxCol) {
    std::size_t tileOldRow = height / kernelSize_h;
    std::size_t tileOldCol = width / (kernelSize_w * N_3D_LAYERS);

    // printf("tileOldRow = %ld / %ld = %ld\n", height, kernelSize_h, tileOldRow);
    // printf("tileOldCol = %ld / (%ld * %d) = %ld\n", width, kernelSize_w, N_3D_LAYERS, tileOldCol);

    int elems_packed = 0;
    uint32_t new_val = 0;

    int tile_size = (SA_H * (SA_W / w_data) * N_3D_LAYERS);
    int nelems_in_oldRow = ((SA_W / w_data) * N_3D_LAYERS);

    // How many columns are produced by all the 3D layers together
    int col_out = (SA_W / w_data) * N_3D_LAYERS;

    // printf("col_out = %d\n", col_out);
    // printf("Tile size = %d\n", tile_size);
    // printf("N elems in oldRow = %d\n", nelems_in_oldRow);

    uint32_t* outPtr = output;

    // Old rows are the new columns
    for (int c = 0; c < tileOldRow; c++){
        // Old columns are the new rows. I store all the new rows first (per each column) 
        for (int r = 0; r < tileOldCol; r++){
            // printf("%d  %d \n", c, r);


            int src_tile_idx = (c * tile_size) + (r * tileOldRow * tile_size);
            // printf("SRC_TILE_IDX = %d\n", src_tile_idx);


            // element in the old row (is the new row element)
            for(int r_elem = 0; r_elem < nelems_in_oldRow; r_elem++){
                for (int w = 0; w < w_data; w++){
                    for (int c_elem = 0; c_elem < SA_H; c_elem++){
                        // printf("--> %d %d\n", r_elem, c_elem);

                        int idx_in_tile = (c_elem * col_out) + r_elem;
                        // int idx_in_tile = (c_elem * N_3D_LAYERS) + r_elem;
                        int src_idx = src_tile_idx + idx_in_tile;

                        // printf("idx in tile = %d\n", idx_in_tile); 
                        // printf("src_idx = %d\n", src_idx); 

                        // uint8_t *bytes = reinterpret_cast<uint8_t*>(&input[src_idx]);
                        // uint8_t byte = bytes[w];

                        uint32_t src_val = input[src_idx];
                        // printf("Src val = %u\n", src_val);

                        uint8_t byte = (src_val >> (8 * (w_data - w - 1))) & 0xFF;

                        // printf("Appending byte[%d] = %d\n", w, static_cast<int8_t>(byte));

                        new_val |= static_cast<uint32_t>(byte) << (8 * (w_data - elems_packed - 1));
                        elems_packed++;
                        // printf("--> %u\n", new_val);


                        if (elems_packed >= w_data){

                            // printf("Adding new val = %u\n", new_val);

                            *outPtr = new_val;
                            outPtr++;
                            new_val = 0;
                            elems_packed = 0;

                            // printf("Added new val = %u\n", *(outPtr-1));
                        }
                    }
                    // printf("\n" );
                }
            }
        }
    }
    // exit(0);
}





void Transpose::multihead_transpose(const uint32_t* input, uint32_t* output, std::size_t seq_len,
                                    std::size_t head_hidden_size, std::size_t num_head) {
    const uint32_t * initial_input = input;
    for (int i=0; i < seq_len; i++){
        for (int n=0; n< num_head; n++){
            input = initial_input + i*head_hidden_size + n*seq_len*head_hidden_size;
            for (int j=0; j < head_hidden_size; j++){
                *output++ = *input++;
            }
        }
    }
}
