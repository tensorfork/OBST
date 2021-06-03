python_flags=`python3-config --cflags --ldflags --includes --libs`
python_flags=`echo "${python_flags//-g}"`
python_include=-I`python3 -c 'import numpy, sys; sys.stdout.write(numpy.get_include()); sys.stdout.flush()'`

optimization_options="-fsingle-precision-constant -fcx-fortran-rules -flto -Ofast -ffast-math -ffinite-math-only -fno-trapping-math -frounding-math -freciprocal-math -fassociative-math -fno-signaling-nans -fstdarg-opt"
code_generation_options="-fwrapv -fPIC -fdelete-dead-exceptions"
preprocessor_options="-pthread"
machine_options="-march=native -mtune=native -msse2 -msse4.2 -shared -mavx -msse4.1 -msse -msse3 -mstackrealign -mmmx -maes -mpclmul -mclflushopt -mfsgsbase -mrdrnd -mf16c -mpopcnt -mfxsr -mxsave -mxsaveopt -msahf -mcx16 -mmovbe -mshstk -mcrc32 -mmwaitx -mrecip -minline-all-stringops"
linker_options="-s -shared"
c_dialect_options="-fopenmp -fopenacc -fsigned-char"

gcc_options="$python_flags $python_include $optimization_options $code_generation_options $preprocessor_options $machine_options $linker_options $c_dialect_options"

gcc_options=`echo $gcc_options | tr '\n' ' ' | tr '\r' ' ' | tr '\t' ' ' | tr '  ' ' '`


echo "Global GCC Flags:"
echo "$gcc_options"


function compile {
  file=${1}
  echo "Cythonizing.."
  cython "$file.pyx"  -3 -Wextra -D
  flags="$file.c $gcc_options -o $file.so"
  echo "Executing gcc.."
  time ((gcc-11 $flags) || (gcc-10 $flags) || (gcc-9 $flags) || (gcc-8 $flags) || (gcc-7 $flags) || (gcc $flags))
  echo "Testing compilation.."
  python3 -c "import $file"
  echo
}
compile train_tokenizer