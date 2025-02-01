/********************************************************
 * Kernels to be optimized for the CS:APP Performance Lab
 ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
/*
 * Please fill in the following team_t struct
 */
team_t team = {

        "e2521680",      /* First student ID */
        "OZAN KAMALI",       /* First student name */

};


/********************
 * NORMALIZATION KERNEL
 ********************/

/****************************************************************
 * Your different versions of the normalization functions go here 
 ***************************************************************/


char new_normalize_descr[] = "My implementation";
void new_normalize(int dim, float *src, float *dst) {
    float min, max;
    float current;
    float difference;
    min = src[0];
    max = src[0];
    int ni = 0;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            current = src[ni + j]; // access once for every element 
            if (current < min) { 
                min = current;
                continue;  // if true dont check other if
            }
            if (current > max) {
                max = current;
            }
        }
        ni += dim;
    }
    ni = 0;
    difference = max - min;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j+=4) {
            dst[ni + j] = (src[ni + j] - min) / (difference);
            dst[ni + j + 1] = (src[ni + j + 1] - min) / (difference);
            dst[ni + j + 2] = (src[ni + j + 2] - min) / (difference);
            dst[ni + j + 3] = (src[ni + j + 3] - min) / (difference);
        }
        ni += dim;
    }
}









 /*
 * naive_normalize - The naive baseline version of convolution
 */
char naive_normalize_descr[] = "naive_normalize: Naive baseline implementation";
void naive_normalize(int dim, float *src, float *dst) {
    float min, max;
    min = src[0];
    max = src[0];

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
	
            if (src[RIDX(i, j, dim)] < min) {
                min = src[RIDX(i, j, dim)];
            }
            if (src[RIDX(i, j, dim)] > max) {
                max = src[RIDX(i, j, dim)];
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            dst[RIDX(i, j, dim)] = (src[RIDX(i, j, dim)] - min) / (max - min);
        }
    }
}

/*
 * normalize - Your current working version of normalization
 * IMPORTANT: This is the version you will be graded on
 */
char normalize_descr[] = "Normalize: Current working version";
void normalize(int dim, float *src, float *dst)
{
    //naive_normalize(dim,src,dst);
    new_normalize(dim, src, dst);
}

/*********************************************************************
 * register_normalize_functions - Register all of your different versions
 *     of the normalization functions  with the driver by calling the
 *     add_normalize_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.
 *********************************************************************/

void register_normalize_functions() {
    add_normalize_function(&naive_normalize, naive_normalize_descr);
    add_normalize_function(&new_normalize, new_normalize_descr);
    add_normalize_function(&normalize, normalize_descr);
    /* ... Register additional test functions here */
    
}




/************************
 * KRONECKER PRODUCT KERNEL
 ************************/

/********************************************************************
 * Your different versions of the kronecker product functions go here
 *******************************************************************/

/*
 * naive_kronecker_product - The naive baseline version of k-hop neighbours
 */





char new2_kronecker_product_descr[] = "New2 Kronecker Product: My baseline implementation";
void new2_kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod) {
    float mult;
    int dim2s = dim2 * dim2;             
    int dim1_dim2s = dim1 * dim2s;       
    int idx_update = (dim1 * dim2) - (dim2 - 1); 
    int base_i, base_ij, idx;          
    int r;                       
    for (int i = 0; i < dim1; i++) {
        base_i = i * dim1_dim2s;        
        for (int j = 0; j < dim1; j++) {
            base_ij = base_i + j * dim2; 
            mult = mat1[RIDX(i, j, dim1)];
            idx = base_ij;              
            r = 0;             
            for (int k = 0; k < dim2s; k++) { // all elements of mat2 * mult into specific prod block
                prod[idx] = mult * mat2[k];
                r++;
                if (r == dim2) {        
                    idx += idx_update;  
                    r = 0;
                } else {
                    idx++;              
                }
            }
        }
    }
}





char new_kronecker_product_descr[] = "My implementation";
void new_kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod) {
    float mult;                        
    int idx = 0;
    int ni = 0;
    for (int i = 0; i < dim1; i++) {
        int nd = 0;
        for (int d = 0; d < dim2; d++) {
            for (int j = 0; j < dim1; j++) { 
                mult = mat1[ni + j]; 
                for (int k = 0; k < dim2; k += 4) {     
                    float val1 = mult * mat2[nd + k];
                    float val2 = mult * mat2[nd + k + 1];
                    float val3 = mult * mat2[nd + k + 2];
                    float val4 = mult * mat2[nd + k + 3];

                    prod[idx] = val1;   // idx goes sequentially instead  
                    prod[idx + 1] = val2;
                    prod[idx + 2] = val3;     
                    prod[idx + 3] = val4;

                    idx += 4;            
                }
            }
            nd += dim2;
        }
        ni += dim1;
    }
}



char naive_kronecker_product_descr[] = "Naive Kronecker Product: Naive baseline implementation";
void naive_kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                for (int l = 0; l < dim2; l++) {
                    prod[RIDX(i, k, dim2) * (dim1 * dim2) + RIDX(j, l, dim2)] = mat1[RIDX(i, j, dim1)] * mat2[RIDX(k, l, dim2)];
                }
            }
        }
    }
}



/*
 * kronecker_product - Your current working version of kronecker_product
 * IMPORTANT: This is the version you will be graded on
 */
char kronecker_product_descr[] = "Kronecker Product: Current working version";
void kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod)
{
    // naive_kronecker_product(dim1,dim2,mat1,mat2,prod);
    // new_kronecker_product(dim1, dim2, mat1, mat2, prod); 2.7
    new_kronecker_product(dim1, dim2, mat1, mat2, prod);
}

/******************************************************************************
 * register_kronecker_product_functions - Register all of your different versions
 *     of the kronecker_product with the driver by calling the
 *     add_kronecker_product_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 ******************************************************************************/

void register_kronecker_product_functions() {
    add_kronecker_product_function(&naive_kronecker_product, naive_kronecker_product_descr);
    add_kronecker_product_function(&new2_kronecker_product, new2_kronecker_product_descr);
    add_kronecker_product_function(&new_kronecker_product, new_kronecker_product_descr);
    add_kronecker_product_function(&kronecker_product, kronecker_product_descr);
    /* ... Register additional test functions here */
}

