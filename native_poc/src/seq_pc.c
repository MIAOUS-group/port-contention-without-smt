#include "seq_pc.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>


int timingSeqPC() {
  char path[100];
  sprintf(path, "./timings_pc.bin");
  size_t ret;
  ret = 0;
  uint64_t *timings = (uint64_t *)calloc(NUM_TIMINGS, sizeof(uint64_t));
  assert(timings != NULL);
  for(int i = 0; i < 1000000000; i++) asm volatile("nop");
  seq_pc(timings);
  FILE *fp;
  fp = fopen(path, "wb");
  assert(fp != NULL);
  ret = fwrite(timings, sizeof(uint64_t), NUM_TIMINGS, fp);
  assert(ret == NUM_TIMINGS);
  fclose(fp);
  free(timings);
//   sleep(1);
  return 0;
}

int timingNoPC() {
  char path[100];
  sprintf(path, "./timings_nopc.bin");
  size_t ret;
  ret = 0;
  uint64_t *timings = (uint64_t *)calloc(NUM_TIMINGS, sizeof(uint64_t));
  assert(timings != NULL);
  for(int i = 0; i < 1000000000; i++) asm volatile("nop");
  noseq_pc(timings);
  FILE *fp;
  fp = fopen(path, "wb");
  assert(fp != NULL);
  ret = fwrite(timings, sizeof(uint64_t), NUM_TIMINGS, fp);
  assert(ret == NUM_TIMINGS);
  fclose(fp);
  free(timings);
//   sleep(1);
  return 0;
}

int main(){
  timingSeqPC();
  timingNoPC();
  return 0;
}
