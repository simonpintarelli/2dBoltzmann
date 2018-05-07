set(CMAKE_REQUIRED_FLAGS "-mavx")
check_cxx_source_runs(
  "
    #include <immintrin.h>
    class VectorizedArray
    {
    public:
      VectorizedArray &
      operator += (const VectorizedArray &vec)
      {
        data = _mm256_add_pd (data, vec.data);
        return *this;
      }
      __m256d data;
    };
    inline
    VectorizedArray
    operator + (const VectorizedArray &u, const VectorizedArray &v)
    {
      VectorizedArray tmp = u;
      return tmp+=v;
    }
    int main()
    {
      __m256d a, b;
      const unsigned int vector_bytes = sizeof(__m256d);
      const int n_vectors = vector_bytes/sizeof(double);
      __m256d * data =
        reinterpret_cast<__m256d*>(_mm_malloc (2*vector_bytes, vector_bytes));
      double * ptr = reinterpret_cast<double*>(&a);
      ptr[0] = (volatile double)(1.0);
      for (int i=1; i<n_vectors; ++i)
        ptr[i] = 0.0;
      b = _mm256_set1_pd ((volatile double)(2.25));
      data[0] = _mm256_add_pd (a, b);
      data[1] = _mm256_mul_pd (b, data[0]);
      ptr = reinterpret_cast<double*>(&data[1]);
      unsigned int return_value = 0;
      if (ptr[0] != 7.3125)
        return_value = 1;
      for (int i=1; i<n_vectors; ++i)
        if (ptr[i] != 5.0625)
          return_value = 1;
      VectorizedArray c, d, e;
      c.data = b;
      d.data = b;
      e = c + d;
      ptr = reinterpret_cast<double*>(&e.data);
      for (int i=0; i<n_vectors; ++i)
        if (ptr[i] != 4.5)
          return_value = 1;
      _mm_free (data);
      return return_value;
    }
    "
  HAVE_AVX)

set(CMAKE_REQUIRED_FLAGS "-mfma")
check_cxx_source_runs(
  "
  #include <immintrin.h>

  int
  main()
  {
    __m256d a, b;
    const unsigned int vector_bytes = sizeof(__m256d);
    const int n_vectors = vector_bytes / sizeof(double);
    __m256d *data = reinterpret_cast<__m256d *>(_mm_malloc(2 * vector_bytes, vector_bytes));
    double *ptr = reinterpret_cast<double *>(&a);
    ptr[0] = (volatile double)(1.0);
    for (int i=1; i<n_vectors; ++i)
      ptr[i] = 0.0;
    b = _mm256_set1_pd ((volatile double)(2.25));
    data[0] = _mm256_add_pd (a, b);
    data[1] = _mm256_fmadd_pd (a, b, data[0]);
    ptr = reinterpret_cast<double*>(&data[1]);
    _mm_free (data);

    return 0;
  }
"
  HAVE_FMA)
