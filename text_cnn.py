from read_files import *
from data_processing import *
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
import tarfile


def text_cnn(maxlen=150, max_features=2000, embed_size=32):
    # Inputs
    comment_seq = Input(shape=[maxlen], name='x_seq')

    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=1, activation='sigmoid')(output)

    model = Model([comment_seq], output)
    #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


if __name__=="__main__":
    if not os.path.exists('./aclImdb'):
        tfile = tarfile.open(r'./aclImdb_v1.tar.gz', 'r:gz')  # r;gz是读取gzip压缩文件
        result = tfile.extractall('./')  # 解压缩文件到当前目录中
    train_texts, train_labels = read_files('train')
    test_texts, test_labels = read_files('test')
    x_train, y_train, x_test, y_test = preprocessing(train_texts, train_labels, test_texts, test_labels)
    model = text_cnn()
    batch_size = 128
    epochs = 20
    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
