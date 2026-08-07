//
// Created by alireza on 4/24/23.
//

#define W_DATA 4
#define KERNEL_DIM SA_SIZE
#define MAX_COL (SA_SIZE/W_DATA)

#include "debuggerFunctions.h"

void print_weight(uint32_t* kernel, int n_row, int n_col){
    for (int i=0; i< n_row; i++){
        for (int j=0; j<n_col; j++){
            printf("0x%08x,\t", kernel[i*n_col + j]);
        }
        printf("\n");
    }
}
void write_weight_to_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            file.write(reinterpret_cast<const char*>(&kernel[i * n_col + j]), sizeof(uint32_t));
        }
    }

    file.close();
}

void read_weight_from_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            file.read(reinterpret_cast<char*>(&kernel[i * n_col + j]), sizeof(uint32_t));
        }
    }

    file.close();
}



void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col){
    uint32_t* initialRowWise = rowWise;
    for (int col=0; col<n_col/MAX_COL; col++){
        rowWise = initialRowWise + col * MAX_COL;
        for (int row=0; row < n_row; row++){
            for (int i=0; i<MAX_COL; i++){
                *(rowWise + i) = *(blockWise+ i);
            }
            blockWise += MAX_COL;
            rowWise += n_col;
        }
    }
}

void rowWise2BlockWise(const uint32_t* rowWise, uint32_t* blockWise, int n_row, int n_col) {
    const uint32_t* initialRowWise = rowWise;
    for (int col = 0; col < n_col / MAX_COL; col++) {
        rowWise = initialRowWise + col * MAX_COL;
        for (int row = 0; row < n_row; row++) {
            for (int i = 0; i < MAX_COL; i++) {
                *(blockWise + i) = *(rowWise + i);
            }
            rowWise += n_col;
            blockWise += MAX_COL;
        }
    }
}


void interleave_hidden_flag(uint32_t* kernel, int n_row, int n_col, uint32_t hidden_flag) {
    for (int i = 0; i < n_row / SA_SIZE; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            int tile_index = (i * (n_col / MAX_COL) + j) * SA_SIZE * MAX_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (all_zeros) {
                kernel[tile_index] = hidden_flag;
            }
        }
    }
}

void interleave_hidden_flag_zero_free(uint32_t*& kernel, int n_row, int n_col, uint32_t hidden_flag) {
    uint32_t * new_kernel;
    new_kernel = new uint32_t [n_row * n_col]();
    uint32_t * new_kernel_ptr = new_kernel;
    int counter = 0;

    for (int i = 0; i < n_row / SA_SIZE; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            int tile_index = (i * (n_col / MAX_COL) + j) * SA_SIZE * MAX_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (!all_zeros) {
                for (int ii = 0; ii < SA_SIZE; ii++) {
                    for (int jj = 0; jj < MAX_COL; jj++) {
                        *new_kernel ++ = kernel[tile_index + ii * MAX_COL + jj];
                    }
                }
            }
            else{
                *new_kernel ++ = hidden_flag;
                counter ++;
            }
        }
    }
    kernel = new_kernel_ptr;
}




void blockWise2Block3Dnano(const uint32_t * blockWise, uint32_t* block3Dnano, int n_row, int n_col){

    int blockRows = n_row / SA_H;
    int blockCols = (n_col * W_DATA) / SA_W;
    int blockSize = SA_H * MAX_COL;
    int groupSize = N_3D_LAYERS;

    // printf("blockRows = %d\n", blockRows);
    // printf("blockCols = %d\n", blockCols);
    // printf("blockSize = %d\n", blockSize);
    // printf("groupSize = %d\n", groupSize);


    uint32_t *dst = block3Dnano;

    for (int groupCol = 0; groupCol < blockCols; groupCol += groupSize) {

        int colsInGroup = std::min(groupSize, blockCols - groupCol);

        for (int blockRow = 0; blockRow < blockRows; blockRow++) {

            for (int c = 0; c < colsInGroup; c++) {

                const uint32_t *src =
                    blockWise +
                    ((groupCol + c) * blockRows + blockRow) * blockSize;

                std::copy(src, src + blockSize, dst);
                dst += blockSize;
            }
        }
    }
}



// void blockWise2Block3Dnano_inputs(const uint32_t * blockWise, uint32_t* block3Dnano, int n_row, int n_col){

//     int blockRows = n_row / SA_H;
//     int blockCols = (n_col * W_DATA) / SA_W;
//     int blockSize = SA_H * MAX_COL;
//     int groupSize = N_3D_LAYERS;

//     printf("blockRows = %d\n", blockRows);
//     printf("blockCols = %d\n", blockCols);
//     printf("blockSize = %d\n", blockSize);
//     printf("groupSize = %d\n", groupSize);


//     uint32_t *dst = block3Dnano;

//     for (int groupCol = 0; groupCol < blockCols; groupCol += groupSize) {

//         int colsInGroup = std::min(groupSize, blockCols - groupCol);

//         for (int blockRow = 0; blockRow < blockRows; blockRow++) {

//             for (int c = 0; c < colsInGroup; c++) {

//                 const uint32_t *src =
//                     blockWise +
//                     ((groupCol + c) * blockRows + blockRow) * blockSize;

//                 std::copy(src, src + blockSize, dst);
//                 dst += blockSize;
//             }
//         }
//     }
// }


// void blockWise2Block3Dnano_inputs(const uint32_t *src,
//                   uint32_t *dst,
//                   int n_rows,
//                   int n_cols)
// {

//     int oldMC = MAX_COL;
//     int newMC = SA_H / W_DATA;

//     int logicalCols = (n_cols * oldMC) / newMC;

//     printf("new = %d\n", newMC);
//     printf("old = %d\n", oldMC);
//     printf("logicalCols = %d\n", logicalCols);


//     for (int group = 0; group < logicalCols; group++) {

//         for (int row = 0; row < n_rows; row++) {

//             for (int k = 0; k < newMC; k++) {


//                 int oldCol = group * newMC + k;
//                 int old_group = n_cols / oldMC;
//                 int old_offset = n_cols % oldMC;

//                 int src_idx =   old_group * n_rows * oldMC +
//                                 row * oldMC +
//                                 old_offset;

//                 int dst_idx = (group * n_rows + row) * newMC + k;

//                 dst[dst_idx] = src[src_idx];
//                 // dst[(group * seq_len + row) * newMC + k] =
//                 //     src[oldCol * seq_len + row];

//                 // printf("\nIT = %d x %d x %d\n", group, row, k);
//                 // printf("SRC idx = %d * %d + %d = %d\n", oldCol, seq_len, row, oldCol * seq_len + row);
//             }
//         }
//     }
// }


void blockWise2Block3Dnano_inputs(const uint32_t *src,
                                  uint32_t *dst,
                                  int rows,
                                  int cols)
{
    int oldMC = MAX_COL;       // current packing
    int newMC = SA_H / W_DATA; // desired packing

    printf("oldMC = %d\n", oldMC);
    printf("newMC = %d\n", newMC);

    // Number of column groups after repacking
    int groups = (cols * oldMC) / newMC;

    printf("rows = %d\n", rows);
    printf("cols = %d\n", cols);
    printf("groups = %d\n", groups);


    for (int group = 0; group < groups; group++) {

        for (int row = 0; row < rows; row++) {

            for (int k = 0; k < newMC; k++) {

                /*
                 * Column index in the original matrix
                 */
                int col = group * newMC + k;


                /*
                 * Source layout:
                 *
                 * oldMC columns are packed together.
                 *
                 * Example oldMC=1:
                 * src[col*rows + row]
                 *
                 * Example oldMC=2:
                 * src[(col/2)*rows*2 + row*2 + (col%2)]
                 */
                int old_group = col / oldMC;
                int old_offset = col % oldMC;

                int src_idx =
                    old_group * rows * oldMC +
                    row * oldMC +
                    old_offset;


                /*
                 * Destination layout:
                 *
                 * group
                 *   row
                 *     packed columns
                 */
                int dst_idx =
                    (group * rows + row) * newMC + k;

                dst[dst_idx] = src[src_idx];
                
            }
        }
    }
}




// void blockWise2Block3Dnano(const uint32_t *blockWise,
//                            uint32_t *block3Dnano,
//                            int n_row,
//                            int n_col)
// {


//     int blockRows = n_row / SA_H;
//     int blockCols = (n_col * W_DATA) / SA_W;
//     int blockSize = SA_H * MAX_COL;



//     int HG = N_3D_LAYERS;
//     int VG = blockRows;

//     printf("blockRows   =   %d\n", n_row / SA_H);
//     printf("blockCols   =   %d\n", (n_col * W_DATA) / SA_W);
//     printf("blockSize   =   %d\n", SA_H * MAX_COL);
//     printf("HG   =   %d\n", N_3D_LAYERS);
//     printf("VG   =   %d\n", blockRows);

//     printf("\n============\n");



//     uint32_t *dst = block3Dnano;


//     for (int row = 0; row < groupRow; row++){

//     }



//     for (int groupCol = 0; groupCol < blockCols; groupCol += HG) {

//         int colsInGroup = std::min(HG, blockCols - groupCol);

//         // printf("GC = %d\n", groupCol);
//         // printf("colsInGroup = %d\n", colsInGroup);

//         for (int groupRow = 0; groupRow < blockRows; groupRow += VG) {

//             int rowsInGroup = std::min(VG, blockRows - groupRow);


//             // printf("GR = %d\n", groupRow);
//             // printf("rowsInGroup = %d\n", rowsInGroup);

//             for (int c = 0; c < colsInGroup; c++) {

//                 for (int r = 0; r < rowsInGroup; r++) {


//                     // printf("[c x r] = %d x %d\n", c, r);

//                     const uint32_t *src =
//                         blockWise +
//                         ((groupCol + c) * blockRows +
//                          (groupRow + r)) * blockSize;

//                     // printf("IDX = %d\n", ((groupCol + c) * blockRows + (groupRow + r)) * blockSize);

//                     // for(int t=0; t<blockSize; t++){
//                     //     printf("%d\n", *(src+t));
//                     // }

//                     std::copy(src, src + blockSize, dst);
//                     dst += blockSize;
//                 }
//             }
//         }
//     }
// }

