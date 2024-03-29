/* Copyright (C) ATHENA AUTHORS
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/platform/logging.h"
using namespace tensorflow;
#ifndef __ADD_RIR_NOISE_AECRES_H_
#define __ADD_RIR_NOISE_AECRES_H_

#ifdef __cplusplus
extern "C" {
#endif

void* add_rir_noise_aecres_init(int nFs);

int add_rir_noise_aecres_process(void* st, short* inputdata,
                                 int inputdata_length, short* outputdata,
                                 int* outputdata_size, bool if_add_rir,
                                 char* rir_filelist, bool if_add_noise,
                                 char* noise_filelist, float snr_min,
                                 float snr_max, bool if_add_aecres,
                                 char* aecres_filelist);

void add_rir_noise_aecres_exit(void* st);

#ifdef __cplusplus
}
#endif

class audio {
 private:
  void* st;

 public:
  audio(int nFs);
  ~audio();

  int audio_pre_proc(short* inputdata, int inputdata_length, short* outputdata,
                     int* outputdata_size, bool if_add_rir, char* rir_filelist,
                     bool if_add_noise, char* noise_filelist, float snr_min,
                     float snr_max, bool if_add_aecres, char* aecres_filelist);
};

#endif  //__ADD_RIR_NOISE_AECRES_H_
