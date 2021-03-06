CS 179 Set 2
Due Wednesday 4/15/2015 @ 3PM.

Put all answers in a file called README.txt.
After answering all of the questions, list how long part 1 and part 2 took.
Feel free to leave any other feedback.

Submit completed sets by emailing to emartin@caltech.edu with subject
"CS 179 Set 2 Submission - Name" where "Name" is your name. Attach to the email
a single archive file (.zip, .tar, .gz, .tar.gz/.tgz) with your README file and
all code.

PART 1
Question 1.1: Latency Hiding (5 points)
---------------------------------------
Approximately how many arithmetic instructions does it take to hide the latency
of a single arithmetic instruction on a GK110?
Assume all of the arithmetic instructions are independent (ie have no
instruction dependencies).
You do not need to consider the number of execution cores on the chip.

Hint: What is the latency of an arithmetic instruction? How many instructions
can a GK110 begin issuing in 1 clock cycle (assuming no dependencies)?

For a GK110, GPU arithmetic instruction latency is about 10 ns. The GK110 has
4 warp schedulers and 2 dispatchers each. This means that in one clock cycle 
the GK110 can start up to 8 instructions at once. The minimum number of 
arithmetic instructions is doing one instruction per clock cycle, so  We would 
see 10 instructions before we dont see latency
anymore. The maximum number of arithmetic instructions is 8 per clock cycle, 
so the maximum number of arithmetic instructions until we hide the latency is
8 per ns * 10 ns = 80 instructions.
by the 
time the first instruction is finished, the latency period is over and so
we can continuously perform arithmetic instructions per clock cylce and not
see the latency of 10ns.


Question 1.2: Thread Divergence (6 points)
------------------------------------------
Let the block shape be (32, 32, 1).

(a)
int idx = threadIdx.y + blockSize.y * threadIdx.x;
if (idx % 32 < 16) {
    foo();
} else {
    bar();
}

Does this code diverge? Why or why not?

No the code will not diverge. Within a given warp, all the threads will have the
same threadIdx.y value, but different threadIdx.x values. This means that
all the idx values % 32 will be the same because the offset from 32 (threadIdx.y)
is the same. Effectively the result will be that all the threads will either
do foo or either bar. Thus, the code will not diverge.

(b)
const float pi = 3.14;
float result = 1.0;
for (int i = 0; i < threadIdx.x; i++) {
    result *= pi;
}

Does this code diverge? Why or why not?
(This is a bit of a trick question, either "yes" or "no can be a correct answer
with appropriate explanation).


Yes, diverges:
Threads will the values threadIdx.x [0,1,2,...,31], the code will diverge.
For the first iteration, we will have all the threads doing the same 
"result *= pi;" instruction, but starting with the next iteration when i is 
incremented, the first thread which has threadIdx.x = 0. Thus, the first
thread diverges from the rest of the threads by not executing the "result *= pi"
instruction. The trend continues as i is incremented, with each iteration, 
another thread will diverge and not execute the same multiply instruction.


Question 1.3: Coalesced Memory Access (9 points)
------------------------------------------------
Let the block shape be (32, 32, 1).
Let data be a (float *) pointing to global memory and let data be 128 byte
aligned (so data % 128 == 0).

Consider each of the following access patterns.

(a)
data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

Yes, the write is coalesced. In a given warp, we have that all the threads
will have the same threadIdx.y value, but varying threadIdx.x values. This means
that in a given warp, this will access memory addresses that are all within a
32 indicies of each other. Each element is a float and so this 32 indicies 
streches over 32 * 4 bytes = 128 bytes. Since we are only accessing values
within this 128 byte cunch, we are only writing to one cache line.

(b)
data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

No, the write is not coalesced. In a given warp, we have that all the threads
will have the same threadIdx.y value, but varying threadIdx.x values. In a 
given warp with threadIdx.y of "i", we will write to data[i + 32], data[i + 
32 * 2], data[i + 32 * 3], ... Each cacheline corresponds to 32 elements, so
we are writing to 32 cache lines, one each thread.

(c)
data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

No, the write is not coalesced. In a given warp, we have that all the threads
will have the same threadIdx.y value, but varying threadIdx.x values. Within
a given threadIdx.y row, we will access the elements [1, 2, 3, ..., 31], 
not the 0 index because we add one. This forces us to go into the first 
element of the next row, and thereby accessing a second cache line. This writes
to two cache lines

Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)
---------------------------------------------------------------------
Let's consider multiplying a 32 x 128 matrix with a 128 x 32
element matrix. This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads
and each thread will compute 1 output element.
Although its not optimal, for the sake of simplicity let's use a single block,
so grid shape = (1, 1, 1), block shape = (32, 32, 1).

For the sake of this problem, let's assume both the left and right matrices have
already been stored in shared memory are in column major format. This means
element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
left hand side and rhs[i + 128 * j] for the right hand side.

This kernel will write to a variable called output stored in shared memory.

Consider the following kernel code:

int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}

(a)
Are there bank conflicts in this code? If so, how many ways is the bank conflict
(2-way, 4-way, etc)?

There are no bank conflicts in the code. Within a given warp, we will have
the same threadIdx.y (j), but different threadIdx.x (i) values for all the threads.
If we analyze all the index accessing patterns of the arrays:

"i + 32 * j" 
j is constant, but i is stride one. Thus, we are accessing the bank in stride
one. This is not a conflict. 

"i + 32 * k" or "i + 32 * (k + 1)"
k changes, but is constant within a given warp. i changes but is stride one. 
Thus, we are accessing the bank in stride one. This is not a conflict.

"k + 128 * j" or "(k + 1) + 128 * j"
k changes, but is constant within given warp. j is constant. Thus, all the 
threads are accessing the same element. This is not a conflict. 

(b)
Expand the inner part of the loop (below)

output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];

into "psuedo-assembly" as was done in the coordinate addition example in lecture
4.

There's no need to expand the indexing math, only to expand the loads, stores,
and math. Notably, the operation a += b * c can be computed by a single
instruction called a fused multiply add (FMA), so this can be a single
instruction in your "psuedo-assembly".

Hint: Each line should expand to 5 instructions.

1. out0 = output[i + 32 * j];
2. left0 = lhs[i + 32 * k];
3. right0 = rhs[k + 128 * j];
4. FMA(out0, left0, right0) // FMA(a, b, c) out0 += left0 + right0
5. output[i + 32 * j] = out0

6. out1 = output[i + 32 * j];
7. left1 = lhs[i + 32 * (k + 1)];
8. right1 = rhs[(k + 1) + 128 * j];
9. FMA(out1, left1, right1) // FMA(a, b, c)
10. output[i + 32 * j] = out1

(c)
Identify pairs of dependent instructions in your answer to part b.

FMA depends on the loading of the variables. This is because we have to know
the values of the variables before we do any computation with them

	4 depends on 1,
	4 depends on 2,
	4 depends on 3,
	9 depends on 6,
	9 depends on 7,
	9 depends on 8

Writing depends on computation of FMA. We have to know the value of FMA before
we write it

	5 depends on 4,
	10 depends on 9

(d)
Rewrite the code given at the beginning of this problem to minimize instruction
dependencies. You can add or delete instructions (deleting an instruction is a
valid way to get rid of a dependency!) but each iteration of the loop must still
process 2 values of k.

int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
  out0 = output[i + 32 * j];
  left0 = lhs[i + 32 * k];
  right0 = rhs[k + 128 * j];
  out0 += left0 * right0 // FMA
  left1 = lhs[i + 32 * (k + 1)];
  right1 = rhs[(k + 1) + 128 * j];
  out0 += left1 * right1; // FMA
  output[i + 32 * j] = out0
}

1. out0 = output[i + 32 * j];
2. left0 = lhs[i + 32 * k];
3. right0 = rhs[k + 128 * j];
4. FMA(out0, left0, right0) // FMA(a, b, c)
5. left1 = lhs[i + 32 * (k + 1)];
6. right1 = rhs[(k + 1) + 128 * j];
7. FMA(out0, left1, right1) // FMA(a, b, c)
8. output[i + 32 * j] = out0

To minimize the instruction dependencies, we get rid of an intermediate write
and an intermediate load. Thus, there are two less dependencies. We can get
rid of the write (previously at line 5) because we are updating the value 
further and writing to memory at the end. We can get rid of the load because
we already have the value of out0 in the register, so we do not have to re-
load it.

(e)
Can you think of any other anything else you can do that might make this code
run faster?

1. out0 = output[i + 32 * j];
2. left0 = lhs[i + 32 * k];
3. left1 = lhs[i + 32 * (k + 1)];
4. right0 = rhs[k + 128 * j];
5. right1 = rhs[(k + 1) + 128 * j];
6. FMA(out0, left0, right0) // FMA(a, b, c)
7. FMA(out0, left1, right1) // FMA(a, b, c)
8. output[i + 32 * j] = out0

We can take advantage of instruction level parallelism to make the code run 
faster. Having all the loads next to each other will allow for parallel loads
into the registers. We also took advantage of spatial locality using the cache.
We switched the order so the lhs array is being accessed sequentially and
same with rhs array. We can also unroll the for loop so we dont have to
do another comparison.

================================================================================

PART 2 - Matrix transpose optimization (65 points)
Optimize the CUDA matrix transpose implementations in transpose_cuda.cu.
Read ALL of the TODO comments. Matrix transpose is a common exercise in GPU
optimization, so do not search for existing GPU matrix transpose code on the
internet.

Your transpose code only need to be able to transpose square matrices where
the side length is a multiple of 64.

The initial implementation has each block of 1024 threads handle a 64x64
block of the matrix, but you can change anything about the kernel if it helps
obtain better performance.

The main method of transpose.cc already checks for correctness for all transpose
results, so there should be an assertion failure if your kernel produces incorrect
output.

The purpose of the shmemTransposeKernel is to demonstrate proper usage of
global and shared memory. The optimalTransposeKernel should be built on top of
shmemTransposeKernel and should incorporate any "tricks" such as ILP, loop unrolling,
vectorized IO, etc that have been discussed in class.

You can compile and run the code by running

make transpose
./transpose

and the build process was tested on minuteman. If this does not work on minutman
for you, be sure to add the lines

export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH

to your ~/.profile file (and then exit and ssh back in to restart your shell).

The transpose program takes 2 optional arguments: input size and method.
Input size must be one of -1, 512, 1024, 2048, 4096, and method must be one
all, cpu, gpu_memcpy, naive, shmem, optimal.
Input size is the first argument and defaults to -1. Method is the second
argument and defaults to all. You can pass input size without passing method,
but you cannot pass method without passing an input size.

Examples:
./transpose
./transpose 512
./transpose 4096 naive
./transpose -1 optimal

Copy paste the output of ./texpranspose.cc into README.txt once you are done.
Describe the strategies used for performance in either block comments over the
kernel (as done for naiveTransposeKernel) or in README.txt.

Output (Optimizations are explained in transpose_cuda.cu):

Size 512 naive CPU: 0.297600 ms
Size 512 GPU memcpy: 0.032160 ms
Size 512 naive GPU: 0.093088 ms
Size 512 shmem GPU: 0.029856 ms
Size 512 optimal GPU: 0.025376 ms

Size 1024 naive CPU: 2.992384 ms
Size 1024 GPU memcpy: 0.082528 ms
Size 1024 naive GPU: 0.310368 ms
Size 1024 shmem GPU: 0.091104 ms
Size 1024 optimal GPU: 0.085504 ms

Size 2048 naive CPU: 35.343903 ms
Size 2048 GPU memcpy: 0.263264 ms
Size 2048 naive GPU: 1.169984 ms
Size 2048 shmem GPU: 0.337920 ms
Size 2048 optimal GPU: 0.306016 ms

Size 4096 naive CPU: 156.513565 ms
Size 4096 GPU memcpy: 1.002816 ms
Size 4096 naive GPU: 4.098560 ms
Size 4096 shmem GPU: 1.235584 ms
Size 4096 optimal GPU: 1.168032 ms

================================================================================

BONUS (+5 points, maximum set score is 100 even with bonus)

Mathematical scripting environments such as Matlab or Python + Numpy often
encouraging expressing algorithms in terms of vector operations because they
offer a convenient and performant interface. For instance, one can add
2 n-component vectors (a and b) in Numpy with c = a + b.

This is often implemented with something like the following code:

void vec_add(float *left, float *right, float *out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = left[i] + right[i];
  }
}

Consider the code
a = x + y + z
where x, y, z are n-component vectors.

One way this could be computed would be

vec_add(x, y, a, n);
vec_add(a, z, a, n);

In what ways is this code (2 calls to vec_add) worse than

for (int i = 0; i < n; i++) {
  a[i] = x[i] + y[i] + z[i];
}

? List at least 2 ways (don't need more than a sentence or two for each way).

1. Instruction dependencies. You would need to schedule the second vec_add after
the first vec_add because the result of x + y needs to be stored in a before
we add z to the updated a. This makes it more difficult and slower to 
parallelize.

2. More Reads and Writes to global memory. Assuming we are working with 
global memory here, there are many more global reads and global writes. This is
because we have to write more intermediate writes and reads to a. 