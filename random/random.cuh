#ifndef RANDOM_CUH
#define RANDOM_CUH

__host__ __device__ inline unsigned long long derive_seed(unsigned long long seed, unsigned int id) {
    unsigned long long x = seed ^ ((unsigned long long)id * 14181477777654086739ULL);
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ inline unsigned long long thread_seed(unsigned long long seed) {
    return derive_seed(seed, blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ inline float _rand_uniform(unsigned long long seed) {
    unsigned long long x = seed;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (unsigned int)(x >> 8 & 0x00FFFFFFu) / (float)0x01000000u;
}

__device__ inline int rand_int(unsigned long long seed, int high) {
    unsigned long long x = seed;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (int)((unsigned int)(x >> 8) % (unsigned int)high);
}

__device__ inline float rand_float(unsigned long long seed) {
    float t = _rand_uniform(seed);
    return t;
}

__device__ inline float rand_normal(unsigned long long seed, float mean, float std) {
    float u1 = _rand_uniform(derive_seed(seed, 0)) * (1.0f - 1e-6f) + 1e-6f;
    float u2 = _rand_uniform(derive_seed(seed, 1));
    return mean + std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

#endif