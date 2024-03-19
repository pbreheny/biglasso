
#ifndef BIGLASSO_OMP_H_
#define BIGLASSO_OMP_H_
// OpenMP setup is OS dependent
// if running on a Mac, this assumes most recent version of libomp has been installed via Homebrew to /usr/local/Cellar
#if __APPLE__ 
#include "/usr/local/Cellar/libomp/17.0.3/include/omp.h"
#elif defined(_OPENMP)
#include <omp.h>
#else
#ifndef DISABLE_OPENMP
// use pragma message instead of warning
#pragma message("Warning: OpenMP is not available, "                    \
"project will be compiled into single-thread code. "                    \
"Use OpenMP-enabled compiler to get benefit of multi-threading.")
#endif
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_num_procs() { return 1; }
inline void omp_set_num_threads(int nthread) {}
inline void omp_set_dynamic(int flag) {}
#endif
#endif //BIGLASSO_OMP_H_
