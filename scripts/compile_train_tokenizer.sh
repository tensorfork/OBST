function compile {
  file=${1}
  cython "$file.pyx"  -3 -Wextra -D
  flags="$file.c `python3-config --cflags --ldflags --includes --libs` -I`python3 -c 'import numpy, sys; sys.stdout.write(numpy.get_include()); sys.stdout.flush()'` -fno-lto -pthread -fPIC -fwrapv -pipe -march=native -mtune=native -Ofast -msse2 -msse4.2 -shared -o $file.so"
  echo "Executing gcc with $flags"
  (gcc-9 $flags) || (gcc-8 $flags) || (gcc-7 $flags) || (gcc $flags)
  echo "Testing compilation.."
  python3 -c "import $file"
  echo
}

compile train_tokenizer