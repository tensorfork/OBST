
try:
    from train_tokenizer import train_local, train_stream
except ImportError:
    import os
    os.system("bash compile_train_tokenizer.sh")
    del os
    from train_tokenizer import train_local, train_stream

if __name__ == '__main__':
    train_local()  # train_stream to stream it if less than 2TB memory are available
