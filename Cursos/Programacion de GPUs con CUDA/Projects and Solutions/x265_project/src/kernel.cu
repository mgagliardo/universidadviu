#ifndef __kernel__
#define __kernel__
#define BLOCK_SIZE 256

__constant__ int num_threads_c[1];
__constant__ unsigned char IntraFilterType_c[][35] =	{
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
	{ 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1 }
	};
__constant__ int bits_arr[33] = {
 	  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3
	};

//kernel helper function
__device__ void iintra_pred_ANGLE(uint8_t* pred_block, int dstStride, int width, int mode, bool bFilter, uint8_t * refLeft,uint8_t* refAbove);
__device__ void cal_satd_kernel(int32_t* y_satd_us_d,int mode,uint8_t* pred_block,uint8_t* orig_block,int luma_size, int channel, int stride);
__device__ void satd_sort(int32_t* y_satd_us_d, int32_t* y_satd_d, uint8_t* y_modes_d);
__device__ void intra_pred_PL_kernel(uint8_t *pred_block, uint8_t *ref_above, uint8_t *ref_left, unsigned int blk_size);
__device__ void intra_pred_DC_kernel(uint8_t *pred_block, uint8_t *ref_above, uint8_t *ref_left, unsigned int blk_size, bool is_luma);
__device__ void init_orig_block_kernel(uint8_t *channel_pix_values, uint8_t *orig_block, unsigned int blk_size, int img_width,
						unsigned int fst_rpix_idx, unsigned int fst_cpix_idx);
__device__ void init_ref_pixels_filt_k(uint8_t *ref_above, uint8_t *ref_left,
			 uint8_t *ref_above_filt, uint8_t *ref_left_filt, unsigned int blk_size);


//kernel code
__global__ void HEVC_kernel(uint8_t * refd_a,uint8_t * refd_l,uint8_t * refd_a_filt,uint8_t * refd_l_filt,uint8_t *channel_pixs, uint8_t * predd_res,uint8_t* origd_res, int img_blk_size, bool bluma,
						uint8_t* y_modes_d,int32_t* y_satd_d, bool is_luma, int32_t* y_satd_us_d, int img_width, int num_of_blk_cols){

	int pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(pos < num_threads_c[0])
	{
		//original block
		int fst_rpix_idx = (pos / num_of_blk_cols) * img_blk_size;
		int fst_cpix_idx = (pos % num_of_blk_cols) * img_blk_size;
		init_orig_block_kernel(channel_pixs, origd_res+pos, img_blk_size, img_width, fst_rpix_idx, fst_cpix_idx);

		//filt ref
		init_ref_pixels_filt_k(refd_a+pos, refd_l+pos, refd_a_filt+pos, refd_l_filt+pos, img_blk_size);

		uint8_t * pred_start = predd_res;
		uint8_t * orig_start = origd_res;

		//planar
			uint8_t *above = (img_blk_size>=8 && is_luma) ? refd_a_filt + num_threads_c[0] + pos : refd_a + num_threads_c[0] + pos;
			uint8_t *left = (img_blk_size >= 8 && is_luma) ? refd_l_filt  + num_threads_c[0] + pos: refd_l + num_threads_c[0] + pos;

			intra_pred_PL_kernel(pred_start+pos, above, left, img_blk_size);
			cal_satd_kernel(y_satd_us_d+pos, 0 , pred_start+pos, orig_start+pos, img_blk_size, 0, img_blk_size);
		//dc
			above = refd_a + num_threads_c[0] + pos;
			left  = refd_l + num_threads_c[0] + pos;
			intra_pred_DC_kernel(pred_start+pos, above, left, img_blk_size, is_luma);
			cal_satd_kernel(y_satd_us_d+pos,1 , pred_start+pos, orig_start+pos, img_blk_size, 0, img_blk_size);

		//anguar	
			int iMode;

			for (iMode = 2; iMode <= 34; iMode++)
			{
				uint8_t * pLeft = (is_luma && IntraFilterType_c[bits_arr[img_blk_size]][iMode]) ?  refd_l_filt+pos : refd_l+pos;
				uint8_t * pAbove = (is_luma && IntraFilterType_c[bits_arr[img_blk_size]][iMode]) ? refd_a_filt+pos : refd_a+pos;
			
				uint8_t * dst = pred_start;

				iintra_pred_ANGLE(dst+pos, img_blk_size, img_blk_size, iMode, bluma, pLeft, pAbove); 
				cal_satd_kernel(y_satd_us_d+pos, iMode , pred_start+pos, orig_start+pos, img_blk_size, 0, img_blk_size);
			}
	
		satd_sort(y_satd_us_d+pos, y_satd_d+pos*35, y_modes_d+pos*35);
	}

	return;
}

void init_ref_pixels_filt_k(uint8_t *ref_above, uint8_t *ref_left,
			 uint8_t *ref_above_filt, uint8_t *ref_left_filt, unsigned int blk_size) {

    ref_above_filt[0] = ref_left_filt[0] = (ref_left[num_threads_c[0]] + (ref_left[0]<<1) + ref_above[num_threads_c[0]] + 2) >> 2;
	for(unsigned int i = num_threads_c[0]; i < (blk_size*num_threads_c[0]<<1); i+=num_threads_c[0]) {
		ref_above_filt[i] = (((ref_above[i-num_threads_c[0]] + (ref_above[i]<<1) + ref_above[i+num_threads_c[0]]) +2 ) >> 2);
		ref_left_filt[i] = (((ref_left[i-num_threads_c[0]] + (ref_left[i]<<1) + ref_left[i+num_threads_c[0]]) +2 ) >> 2);
	}
	ref_above_filt[(blk_size*num_threads_c[0]<<1)] = ref_above[(blk_size*num_threads_c[0]<<1)];
	ref_left_filt[(blk_size*num_threads_c[0]<<1)] = ref_left[(blk_size*num_threads_c[0]<<1)];
}


void init_orig_block_kernel(uint8_t *channel_pix_values, uint8_t *orig_block, unsigned int blk_size, int img_width,
						unsigned int fst_rpix_idx, unsigned int fst_cpix_idx) {

	unsigned int p = fst_rpix_idx*img_width + fst_cpix_idx;
	int blk_stride_idx_inc = blk_size*num_threads_c[0];
	int idx_i = 0; 
	for(unsigned int i = 0; i < blk_size; i++) {
		int idx_j = 0;
		for(unsigned int j = 0; j < blk_size; j++) {
			orig_block[idx_i+idx_j] = channel_pix_values[p+j];
			idx_j += num_threads_c[0];
		}
//		memcpy(orig_block+i*blk_size, channel_pix_values+p, blk_size*sizeof(uint8_t));
		p += img_width;
		idx_i += blk_stride_idx_inc;
	}
}

//#define HADAMARD4(d0, d1, d2, d3, s0, s1, s2, s3) { \
//}
///*
#define HADAMARD4(d0, d1, d2, d3, s0, s1, s2, s3) { \
        uint32_t t0 = s0 + s1; \
        uint32_t t1 = s0 - s1; \
        uint32_t t2 = s2 + s3; \
        uint32_t t3 = s2 - s3; \
        d0 = t0 + t2; \
        d2 = t0 - t2; \
        d1 = t1 + t3; \
        d3 = t1 - t3; \
}
//*/

__device__ uint32_t abs_d(uint32_t a)
{
    uint32_t s = ((a >> (16 - 1)) & (((uint32_t)1 << 16) + 1)) * ((uint16_t)-1);

    return (a + s) ^ s;
}
__device__ int satd_4x4_kernel(uint8_t *pix1, uint8_t *pix2, int stride)
{
//	printf("the stride is %d\n",stride);
    uint32_t temp[4][2];
    uint32_t a0, a1, a2, a3, b0, b1;
    uint32_t sum = 0;

    for (int i = 0; i < 4; i++, pix1 += stride, pix2 += stride)
    {
        a0 = pix1[0] - pix2[0];
		int pix_idx = num_threads_c[0];
        a1 = pix1[pix_idx] - pix2[pix_idx];
        b0 = a0 + a1 + ((a0 - a1) << 16);
		pix_idx += num_threads_c[0];
        a2 = pix1[pix_idx] - pix2[pix_idx];
		pix_idx += num_threads_c[0];
        a3 = pix1[pix_idx] - pix2[pix_idx];
        b1 = a2 + a3 + ((a2 - a3) << 16);
        temp[i][0] = b0 + b1;
        temp[i][1] = b0 - b1;
		
    }

//	printf("Stu: a0: %4d, a1: %4d, a2: %4d, a3: %d\n", a0,a1,a2,a3);
    for (int i = 0; i < 2; i++)
    {
        HADAMARD4(a0, a1, a2, a3, temp[0][i], temp[1][i], temp[2][i], temp[3][i]);
        a0 = abs_d(a0) + abs_d(a1) + abs_d(a2) + abs_d(a3);
        sum += (uint16_t)a0 + (a0 >> 16);
    }
/*
	pix1 -= (stride<<2);
	pix2 = pix2 - (stride<<2);
    for (int i = 0; i < 4; i++, pix1 += stride, pix2 += stride)
    {
        a0 = (uint32_t)(pix1[0] - pix2[0]);
		int pix_idx = num_threads_c[0];
        a1 = (uint32_t)(pix1[pix_idx] - pix2[pix_idx]);
        b0 = a0 - a1;
		pix_idx += num_threads_c[0];
        a2 = (uint32_t)(pix1[pix_idx] - pix2[pix_idx]);
		pix_idx += num_threads_c[0];
        a3 = (uint32_t)(pix1[pix_idx] - pix2[pix_idx]);
        b1 = a2 - a3;
        temp[i][0] = b0 + b1;
        temp[i][1] = b0 - b1;

    }

    for (int i = 0; i < 2; i++)
    {
        HADAMARD4(a0, a1, a2, a3, temp[0][i], temp[1][i], temp[2][i], temp[3][i]);
        a0 = abs((int)a0) + abs((int)a1) + abs((int)a2) + abs((int)a3);
        sum += a0;
	
    }*/
        return (int)(sum >> 1);
}


__device__ int _satd_8x8_kernel(uint8_t *pix1, uint8_t *pix2, int stride)
{
    uint32_t temp[8][4];
    uint32_t a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3;
    uint32_t sum = 0;

    for (int i = 0; i < 8; i++, pix1 += stride, pix2 += stride)
    {
        a0 = pix1[0] - pix2[0];
		int pix_idx = num_threads_c[0];
        a1 = pix1[pix_idx] - pix2[pix_idx];
		pix_idx += num_threads_c[0];
        b0 = (a0 + a1) + ((a0 - a1) << 16);
        a2 = pix1[pix_idx] - pix2[pix_idx];
		pix_idx += num_threads_c[0];
        a3 = pix1[pix_idx] - pix2[pix_idx];
		pix_idx += num_threads_c[0];
        b1 = (a2 + a3) + ((a2 - a3) << 16);
        a4 = pix1[pix_idx] - pix2[pix_idx];
		pix_idx += num_threads_c[0];
        a5 = pix1[pix_idx] - pix2[pix_idx];
		pix_idx += num_threads_c[0];
        b2 = (a4 + a5) + ((a4 - a5) << 16);
        a6 = pix1[pix_idx] - pix2[pix_idx];
		pix_idx += num_threads_c[0];
        a7 = pix1[pix_idx] - pix2[pix_idx];
        b3 = (a6 + a7) + ((a6 - a7) << 16);
        HADAMARD4(temp[i][0], temp[i][1], temp[i][2], temp[i][3], b0, b1, b2, b3);
    }

    for (int i = 0; i < 4; i++)
    {
        HADAMARD4(a0, a1, a2, a3, temp[0][i], temp[1][i], temp[2][i], temp[3][i]);
        HADAMARD4(a4, a5, a6, a7, temp[4][i], temp[5][i], temp[6][i], temp[7][i]);
        b0  = abs_d((a0 + a4)) + abs_d((a0 - a4));
        b0 += abs_d((a1 + a5)) + abs_d((a1 - a5));
        b0 += abs_d((a2 + a6)) + abs_d((a2 - a6));
        b0 += abs_d((a3 + a7)) + abs_d((a3 - a7));
        sum += (uint16_t)b0 + (b0 >> 16);
    }/*
	pix1 -= (stride<<3);
	pix2 = pix2 - (stride<<3);
    for (int i = 0; i < 8; i++, pix1 += stride, pix2 += stride)
    {
        a0 = uint32_t(pix1[0] - pix2[0]);
		int pix_idx = num_threads_c[0];
        a1 = uint32_t(pix1[pix_idx] - pix2[pix_idx]);
		pix_idx += num_threads_c[0];
        b0 = (a0 - a1);
        a2 = uint32_t(pix1[pix_idx] - pix2[pix_idx]);
		pix_idx += num_threads_c[0];
        a3 = uint32_t(pix1[pix_idx] - pix2[pix_idx]);
		pix_idx += num_threads_c[0];
        b1 = (a2 - a3);
        a4 = uint32_t(pix1[pix_idx] - pix2[pix_idx]);
		pix_idx += num_threads_c[0];
        a5 = uint32_t(pix1[pix_idx] - pix2[pix_idx]);
		pix_idx += num_threads_c[0];
        b2 = (a4 - a5);
        a6 = uint32_t(pix1[pix_idx] - pix2[pix_idx]);
		pix_idx += num_threads_c[0];
        a7 = uint32_t(pix1[pix_idx] - pix2[pix_idx]);
        b3 = (a6 - a7);

        HADAMARD4(temp[i][0], temp[i][1], temp[i][2], temp[i][3], b0, b1, b2, b3);
    }

    for (int i = 0; i < 4; i++)
    {
        HADAMARD4(a0, a1, a2, a3, temp[0][i], temp[1][i], temp[2][i], temp[3][i]);
        HADAMARD4(a4, a5, a6, a7, temp[4][i], temp[5][i], temp[6][i], temp[7][i]);
        b0  = abs((int)(a0 + a4)) + abs((int)(a0 - a4));
        b0 += abs((int)(a1 + a5)) + abs((int)(a1 - a5));
        b0 += abs((int)(a2 + a6)) + abs((int)(a2 - a6));
        b0 += abs((int)(a3 + a7)) + abs((int)(a3 - a7));
        sum += b0;
    }*/
    return sum;
}

__device__ int satd_8x8_kernel(uint8_t *pix1, uint8_t *pix2, int stride)
{
	return (int)((_satd_8x8_kernel(pix1,pix2,stride)+2)>>2);
}



__device__ int satd_16x16_kernel(uint8_t *pix1, uint8_t *pix2, int stride)
{
    int sum = _satd_8x8_kernel(pix1, pix2, stride)
        + _satd_8x8_kernel(pix1 + 8*num_threads_c[0], pix2 + 8*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1 + 128*num_threads_c[0], pix2 + 128*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1 + 136*num_threads_c[0], pix2 + 136*num_threads_c[0], stride);

    return (int)((sum+2)>>2);
}


__device__ int satd_32x32_kernel(uint8_t *pix1, uint8_t *pix2, int stride)
{
    int sum1 = _satd_8x8_kernel(pix1, pix2, stride)
        + _satd_8x8_kernel(pix1 + 8*num_threads_c[0], pix2 + 8*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1 + 256*num_threads_c[0], pix2 + 256*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1 + 264*num_threads_c[0], pix2 + 264*num_threads_c[0], stride);
	sum1 = ((sum1+2)>>2);

    int sum2 = _satd_8x8_kernel(pix1 + 16*num_threads_c[0], pix2 + 16*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1 + 24*num_threads_c[0], pix2 + 24*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1 + 272*num_threads_c[0], pix2 + 272*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1 + 280*num_threads_c[0], pix2 + 280*num_threads_c[0], stride);
	sum2 = ((sum2+2)>>2);

    int sum3 = _satd_8x8_kernel(pix1+ 512*num_threads_c[0], pix2+ 512*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1+ 520*num_threads_c[0], pix2+ 520*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1+ 768*num_threads_c[0], pix2+ 768*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1+ 776*num_threads_c[0], pix2+ 776*num_threads_c[0], stride);
	sum3 = ((sum3+2)>>2);

    int sum4 = _satd_8x8_kernel(pix1+ 528*num_threads_c[0], pix2+ 528*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1+ 536*num_threads_c[0], pix2+536*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1+ 784*num_threads_c[0], pix2+ 784*num_threads_c[0], stride)
        + _satd_8x8_kernel(pix1+ 792*num_threads_c[0], pix2+ 792*num_threads_c[0], stride);
	sum4 = ((sum4+2)>>2);

    return sum1+sum2+sum3+sum4;
}


__device__ void cal_satd_kernel(int32_t* y_satd_us_d, int mode,uint8_t* pred_block,uint8_t* orig_block,int luma_size, int channel, int stride){
	stride *= num_threads_c[0];
	int satd;

	if(luma_size == 4)
	{
		satd = satd_4x4_kernel(pred_block, orig_block, stride);
	}
	else if(luma_size == 8)
	{
		satd = satd_8x8_kernel(pred_block, orig_block, stride);
	}
	else if(luma_size == 16)
	{
		satd = satd_16x16_kernel(pred_block, orig_block, stride);
	}
	else if(luma_size == 32)
	{
		satd = satd_32x32_kernel(pred_block, orig_block, stride);
	}

	y_satd_us_d[mode*num_threads_c[0]] = (satd<<8) | mode;
}


__device__ void satd_sort(int32_t* y_satd_us_d, int32_t* y_satd_d, uint8_t* y_modes_d) {

	int pos;
	int idx_i = 0, idx_j;
	for(int i = 0; i < 35; i++) {
		pos = 0;
		idx_j = 0;
		for(int j = 0; j < 35; j++) {
			if(y_satd_us_d[idx_j] < y_satd_us_d[idx_i]) {
				pos++;                                                                                    
			} // if
			idx_j += num_threads_c[0];
		}

		y_modes_d[pos] = y_satd_us_d[idx_i] & 0xff;
		y_satd_d[pos] = y_satd_us_d[idx_i] >> 8;
		idx_i += num_threads_c[0];
	} //for(i)
}


__device__ int16_t imy_clip(int16_t min, int16_t max, int16_t a)
{
	if (a < min)
		return min;
	else if(a > max)
		return max;
	else
		return a;
}

__device__ void intra_pred_PL_kernel(uint8_t *pred_block, uint8_t *ref_above, uint8_t *ref_left, unsigned int blk_size) {

	unsigned int k, l;
    uint8_t bottomLeft, topRight;
    int horPred;
    // OPT_ME: when width is 64, the shift1D is 8, then the dynamic range is 17 bits or [-65280, 65280], so we have to use 32 bits here
    int32_t leftColumn[MAX_CU_SIZE + 1], topRow[MAX_CU_SIZE + 1];
    // CHECK_ME: dynamic range is 9 bits or 15 bits(I assume max input bit_depth is 14 bits)
    int16_t bottomRow[MAX_CU_SIZE], rightColumn[MAX_CU_SIZE];
    int offset2D = blk_size;


    //int shift1D = g_convertToBit[blk_size] + 2;
	int shift1D = (blk_size==4)? 0: (blk_size==8)? 1: (blk_size==16)? 2:3; 
	shift1D += 2;	

	int shift2D = shift1D + 1;

    // Get left and above reference column and row
	int ref_idx = 0;
    for (k = 0; k < (blk_size+1) ; k++)
    {
//        topRow[k] =ref_above[k*num_threads_c[0]];
        topRow[k] =ref_above[ref_idx];
//        leftColumn[k] =ref_left[k*num_threads_c[0]];
        leftColumn[k] =ref_left[ref_idx];
		ref_idx += num_threads_c[0];
    }

    // Prepare intermediate variables used in interpolation
    bottomLeft = (uint8_t)leftColumn[blk_size];
    topRight   = (uint8_t)topRow[blk_size];
    for (k = 0; k < blk_size; k++)
    {
        bottomRow[k]   = (int16_t)(bottomLeft - topRow[k]);
        rightColumn[k] = (int16_t)(topRight - leftColumn[k]);
        topRow[k]      <<= shift1D;
        leftColumn[k]  <<= shift1D;
		//printf("bot: %d, right: %d, top: %d, left: %d\n",bottomRow[k], rightColumn[k],topRow[k], leftColumn[k]);
    }

    // Generate prediction signal
	int idx_o = 0;
	int stride_idx_inc = blk_size *num_threads_c[0];
    for (k = 0; k < blk_size; k++)
    {
        horPred = leftColumn[k] + offset2D;
		int idx_i = 0;
        for (l = 0; l < blk_size; l++)
        {
            horPred += rightColumn[k];
            topRow[l] += bottomRow[l];
//            pred_block[(k*blk_size+l)*num_threads_c[0]] = (uint8_t)((horPred + topRow[l]) >> shift2D);
            pred_block[idx_o+idx_i] = (uint8_t)((horPred + topRow[l]) >> shift2D);
			idx_i += num_threads_c[0];
        }
		idx_o += stride_idx_inc;
    }
	
}

__device__ void intra_pred_DC_kernel(uint8_t *pred_block, uint8_t *ref_above, uint8_t *ref_left, unsigned int blk_size, bool is_luma) {
	unsigned int w, sum = 0;

	int ref_idx = 0;
    for (w = 0; w < blk_size; w++)
    {
        sum += ref_above[ref_idx];
		sum += ref_left[ref_idx];
		ref_idx += num_threads_c[0];
    }

    uint8_t pred_dc_val = (uint8_t)((sum + blk_size) / (blk_size + blk_size));

	// assign the predict value to all the block pixels
	int pred_idx_o = 0;
	int pred_idx_i;
	int stride_idx_inc = blk_size*num_threads_c[0];
	for(unsigned int r = 0; r < blk_size; r++) {
		pred_idx_i = 0;
		for(unsigned int c = 0; c < blk_size; c++) {
//			pred_block[(r*blk_size+c)*num_threads_c[0]] = pred_dc_val;
			pred_block[pred_idx_i+pred_idx_o] = pred_dc_val;
			pred_idx_i += num_threads_c[0];
		}
		pred_idx_o += stride_idx_inc;
	}

	// FIR filter when needed
	if(blk_size <= 16 && is_luma) {
		unsigned int y, p;

		// boundary pixels processing
		pred_block[0] = (uint8_t)((ref_above[0] + ref_left[0] + (pred_block[0]<<1) + 2) >> 2);

		//ref_idx = num_threads_c[0];
		ref_idx = num_threads_c[0];
		pred_idx_o = stride_idx_inc;
		for (y = 1, p = blk_size; y < blk_size; y++, p += blk_size)
		{
//			pred_block[y*num_threads_c[0]] = (uint8_t)((ref_above[ref_idx] +  3 * pred_block[y*num_threads_c[0]] + 2) >> 2);
			pred_block[ref_idx] = (uint8_t)((ref_above[ref_idx] +  3 * pred_block[ref_idx] + 2) >> 2);
//			pred_block[p*num_threads_c[0]] = (uint8_t)((ref_left[ref_idx] + 3 * pred_block[p*num_threads_c[0]] + 2) >> 2);
			pred_block[pred_idx_o] = (uint8_t)((ref_left[ref_idx] + 3 * pred_block[pred_idx_o] + 2) >> 2);
			ref_idx += num_threads_c[0];
			pred_idx_o += stride_idx_inc;
		}
	}
}

__device__ void iintra_pred_ANGLE(uint8_t* pred_block, int dstStride, int width, int mode, bool bFilter, uint8_t * refLeft,uint8_t* refAbove) 
{
    int k, l;
    int blkSize  = width;

    // Map the mode index to main prediction direction and angle
    bool modeHor       = (mode < 18);
    bool modeVer       = !modeHor;
    //int intraPredAngle = modeVer ? (int)mode - VER_IDX : modeHor ? -((int)mode - HOR_IDX) : 0;
    int intraPredAngle = modeVer ? (int)mode - 26 : modeHor ? -((int)mode - 10) : 0;
    int absAng         = abs(intraPredAngle);
    int signAng        = intraPredAngle < 0 ? -1 : 1;

    // Set bitshifts and scale the angle parameter to block img_blk_size
    int angTable[9]    = { 0,    2,    5,   9,  13,  17,  21,  26,  32 };
    int invAngTable[9] = { 0, 4096, 1638, 910, 630, 482, 390, 315, 256 }; // (256 * 32) / Angle
    int invAngle       = invAngTable[absAng];
    absAng             = angTable[absAng];
    intraPredAngle     = signAng * absAng;

    // Do angular predictions
    {
        uint8_t* refMain;
        uint8_t* refSide;
		int ref_idx;
		int pred_idx_o, pred_idx_i;
		int stride_idx_inc = dstStride*num_threads_c[0];

//printf("mode = %d, bool = %d\n", mode, intraPredAngle<0);

        // Initialise the Main and Left reference array.
        if (intraPredAngle < 0)
        {
            refMain = (modeVer ? refAbove : refLeft); // + (blkSize - 1);
            refSide = (modeVer ? refLeft : refAbove); // + (blkSize - 1);

            // Extend the Main reference to the left.
            int invAngleSum    = 128; // rounding for (shift by 8)
			ref_idx = -num_threads_c[0];
            for (k = -1; k > blkSize * intraPredAngle >> 5; k--)
            {
                invAngleSum += invAngle;
                refMain[ref_idx] = refSide[(invAngleSum>>8)*num_threads_c[0]];
				ref_idx -= num_threads_c[0];
            }
        }
        else
        {
            refMain = modeVer ? refAbove : refLeft;
            refSide = modeVer ? refLeft  : refAbove;
        }

        if (intraPredAngle == 0)
        {
			pred_idx_o = 0;
            for (k = 0; k < blkSize; k++)
            {
				ref_idx = num_threads_c[0];
				pred_idx_i = 0;
                for (l = 0; l < blkSize; l++)
                {
//                    pred_block[(k*dstStride+l)*num_threads_c[0]] = refMain[ref_idx];
                    pred_block[pred_idx_o+pred_idx_i] = refMain[ref_idx];
					ref_idx += num_threads_c[0];
					pred_idx_i += num_threads_c[0];
                }
				pred_idx_o += stride_idx_inc;
            }

            if (bFilter)
            {
				ref_idx = num_threads_c[0];
				pred_idx_o = 0;
                for (k = 0; k < blkSize; k++)
                {
//					pred_block[(k*dstStride)*num_threads_c[0]] = (uint8_t)imy_clip((int16_t)0, (int16_t)((1 << 8) - 1), 
//								static_cast<int16_t>((pred_block[(k*dstStride)*num_threads_c[0]]) + ((refSide[ref_idx] - refSide[0]) >> 1)));
					pred_block[pred_idx_o] = (uint8_t)imy_clip((int16_t)0, (int16_t)((1 << 8) - 1), 
								static_cast<int16_t>((pred_block[pred_idx_o]) + ((refSide[ref_idx] - refSide[0]) >> 1)));
					ref_idx += num_threads_c[0];
					pred_idx_o += stride_idx_inc;
                }
            }
        }
        else
        {
            int deltaPos = 0;
            int deltaInt;
            int deltaFract;

			pred_idx_o = 0;
            for (k = 0; k < blkSize; k++)
            {
                deltaPos += intraPredAngle;
                deltaInt   = deltaPos >> 5;
                deltaFract = deltaPos & (32 - 1);

                if (deltaFract)
                {
                    // Do linear filtering
					ref_idx = (deltaInt+1)*num_threads_c[0];
					pred_idx_i = 0;
                    for (l = 0; l < blkSize; l++)
                    {
//                        pred_block[(k*dstStride+l)*num_threads_c[0]] = (uint8_t)(((32 - deltaFract) * refMain[ref_idx] + deltaFract * refMain[ref_idx+num_threads_c[0]] + 16) >> 5);
                        pred_block[pred_idx_o+pred_idx_i] = (uint8_t)(((32 - deltaFract) * refMain[ref_idx] + deltaFract * refMain[ref_idx+num_threads_c[0]] + 16) >> 5);
						ref_idx += num_threads_c[0];
						pred_idx_i += num_threads_c[0];
                    }
                }
                else
                {
                    // Just copy the integer samples
					ref_idx = (deltaInt+1)*num_threads_c[0];
					pred_idx_i = 0;
                    for (l = 0; l < blkSize; l++)
                    {
//                        pred_block[(k*dstStride+l)*num_threads_c[0]] = refMain[ref_idx];
                        pred_block[pred_idx_o+pred_idx_i] = refMain[ref_idx];
						ref_idx += num_threads_c[0];
						pred_idx_i += num_threads_c[0];
                    }
                }
				pred_idx_o += stride_idx_inc;
            }
        }
		//add
        if (modeHor)
        {
            uint8_t  tmp;
			pred_idx_o = 0;
			int a = 0;
			int b = 0;
            for (k = 0; k < blkSize - 1; k++)
            {
				pred_idx_i = (k+1)*num_threads_c[0];
				b = (k+1)*stride_idx_inc;
                for (l = k + 1; l < blkSize; l++)
                {
                    tmp  = pred_block[pred_idx_o+pred_idx_i];

                    pred_block[pred_idx_o+pred_idx_i] = pred_block[a+b];
                    pred_block[a+b] = tmp;

					pred_idx_i += num_threads_c[0];
					b += stride_idx_inc;
//                    tmp  = pred_block[(k*dstStride+l)*num_threads_c[0]];
//                    pred_block[(k*dstStride+l)*num_threads_c[0]] = pred_block[(l*dstStride+k)*num_threads_c[0]];
//                    pred_block[(l*dstStride+k)*num_threads_c[0]] = tmp;
//					pred_idx_i += num_threads_c[0];
                }
				pred_idx_o += stride_idx_inc;
				a += num_threads_c[0];
            }
        }
    }
}

#endif
