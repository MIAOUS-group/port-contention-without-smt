
#ifndef SEQ_PC_H
#define SEQ_PC_H

#define PHY_CORE 4
#define NUM_TIMINGS (1<<10)

#ifndef __ASSEMBLER__
#include <stdint.h>

extern void seq_pc();
extern void noseq_pc();
#endif

#endif
