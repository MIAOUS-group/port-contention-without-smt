#include "seq_pc.h"

.text


# First experiment, two different loops, should create contention
.global seq_pc
.p2align 4
seq_pc:

mov $NUM_TIMINGS, %rcx

1:
lfence
rdtsc # rdx:rax
lfence
mov %rax, %rsi

.rept 500000 #flag
#First instruction, you can modify it
aesdec %xmm0, %xmm1 #instruction 1
.endr
#mfence
.rept 500000 #flag
#Second instruction, you can modify it
popcnt %r8, %r8 #instruction 2
.endr

lfence
rdtsc
shl $32, %rax
or %rsi, %rax
mov %rax, (%rdi)
add $8, %rdi
dec %rcx
jnz 1b
ret



.global noseq_pc
.p2align 4
noseq_pc:

mov $NUM_TIMINGS, %rcx

1:
lfence
rdtsc # rdx:rax
lfence
mov %rax, %rsi

.rept 500000 #flag
aesdec %xmm0, %xmm1 #instruction 1
popcnt %r8, %r8 #instruction 2
.endr

lfence
rdtsc
shl $32, %rax
or %rsi, %rax
mov %rax, (%rdi)
add $8, %rdi
dec %rcx
jnz 1b
ret
