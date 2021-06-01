import os

try:
    from train_tokenizer import main
except ImportError:
    os.system("bash compile_train_tokenizer.sh")
    from train_tokenizer import main

if __name__ == '__main__':
    main()
