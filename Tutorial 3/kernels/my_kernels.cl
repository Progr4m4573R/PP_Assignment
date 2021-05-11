kernel void reduction_complete(__global float4* A, local float4* partial_sums, global float* B) {
 int lid = get_local_id(0);
 int group_size = get_local_size(0);
 partial_sums[lid] = A[get_local_id(0)];

 barrier(CLK_LOCAL_MEM_FENCE);
	 for(int i = group_size/2; i>0; i >>= 1) {
		if(lid < i) {
			partial_sums[lid] += partial_sums[lid + i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
	 }
	 if(lid == 0) {
		*B = partial_sums[0].s0 + partial_sums[0].s1 +
		partial_sums[0].s2 + partial_sums[0].s3;
	 }
}

//fixed 4 step reduce
kernel void reduce_add_1(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying
	 
	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
}

//flexible step reduce 
kernel void reduce_add_2(global const float* A, global float* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//reduce using local memory (so called privatisation)
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

//a very simple histogram implementation
kernel void hist_simple(global const int* A, global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//calculates the block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N && id < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}
//Bitonic sort implementation-----------------------------------------------------------------------------

kernel void cmpxchg(global int* A, global int* B, global bool* dir){
	if((!dir && *A > *B) || (dir && *A < *B)){// false is ascending and true is descending
	int t = *A;
	*A = *B;
	*B = t;
	}
}

kernel void bitonic_merge(int id, global int* A, int N, global bool* dir){//This takes a bitonic sequence
	for(int i = N/2; i > 0; i/=2){
		if((id %(i*2)) < i)
			cmpxchg(&A[id], &A[id+i], dir);

		barrier(CLK_GLOBAL_MEM_FENCE);
		}
}
kernel void sort_bitonic(global int* A){//This makes an unordered list a bitonic sequence
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 1; i < N/2; i*=2){
		if (id % (i*4) < i*2)
			bitonic_merge(id, A, i*2, false);
		else if ((id + i*2) % (i*4) < i*2)
			bitonic_merge(id, A, i*2, true);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	bitonic_merge(id, A, N, false);// This needs to be called otherwise the result will just be a bitonic sequence// Final merge in ascending order fully sorts the data
	
}

//Min and max  implementation-----------------------------------------------------------------
kernel void min_max(global float* A, global float* C){
	float first = A[get_local_id(0)];
	float last = A[get_local_id(-1)];
	C[0] = first;
	C[1] = last;
}

//Variance implementation-----------------------------------------------------------------------------

kernel void my_variance(global const float* A, float mean, int N, int id, global float* D){// mean might be able to be set as local, not sure yet
	//Replace all the values in D with the difference between each value and the mean squared
	for(int i = 1; i < N; i += 1){
		D[id] = (A[id] - mean)*(A[id] - mean); 
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	//sum up the squared differences and put in index 0
	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			D[id] += D[id + i];
	barrier(CLK_GLOBAL_MEM_FENCE);
	}
	//variance sum of squared differences divided by data size and put the variance result in the first index
	D[id] = D[id]/(N-1);
}

kernel void parallel_variance(global const float* A, global float* D){
	int id = get_global_id(0); 
	int N = get_global_size(0); 

	//define variables here so they can be accesses outside of the function
	int mean;

	//Copy contents of A to D
	D[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N))
				D[id] += D[id + i];// store sum in temp
		
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	//so i get the mean by dividing the sum by N, the data size
	mean = D[0]/N;
	// call varaince with mean calculated
	my_variance(A,mean,N,id,D);
	barrier(CLK_GLOBAL_MEM_FENCE);
}

