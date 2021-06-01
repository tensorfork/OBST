
try:
    from train_tokenizer import main
except ImportError:
    import os
    os.system("bash compile_train_tokenizer.sh")
    del os
    from train_tokenizer import main

if __name__ == '__main__':
    main()
