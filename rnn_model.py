# KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras import backend as K
from keras import optimizers
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix, hstack
import gc


def rmsle(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5


dr = 0.25


def get_model():
    # params
    dr_r = dr

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]],
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_size = 60

    emb_name = Embedding(MAX_TEXT, emb_size // 3)(name)
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc)
    emb_category_name = Embedding(MAX_TEXT, emb_size // 3)(category_name)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_category_name)
    rnn_layer3 = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand)
        , Flatten()(emb_category)
        , Flatten()(emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , num_vars
    ])
    main_l = Dropout(0.3)(Dense(512, activation='relu')(main_l))
    main_l = Dropout(0.2)(Dense(88, activation='relu')(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name, item_desc, brand
                      , category, category_name
                      , item_condition, num_vars], output)
    # optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss="mse",
                  optimizer=optimizer)
    return model


def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)

    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: " + str(v_rmsle))
    return v_rmsle


# fin_lr=init_lr * (1/(1+decay))**(steps-1)
exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1

print('[{}] Finished DEFINEING MODEL...'.format(time.time() - start_time))

gc.collect()
# FITTING THE MODEL
epochs = 2
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.009, 0.006
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                       'bs', str(BATCH_SIZE),
                       'lrI', str(lr_init),
                       'lrF', str(lr_fin),
                       'dr', str(dr)])

model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

history = model.fit(X_train, dtrain.target
                    , epochs=epochs
                    , batch_size=BATCH_SIZE
                    , validation_split=0.01
                    # , callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=10
                    )
print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))
# EVLUEATE THE MODEL ON DEV TEST
v_rmsle = eval_model(model)
print('[{}] Finished predicting valid set...'.format(time.time() - start_time))

# CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = np.expm1(preds)
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]][:test_len]
submission["price"] = preds[:test_len] * 0.8
print('[{}] Finished predicting test set...'.format(time.time() - start_time))

del train
del test
gc.collect()

nrow_train = df_train.shape[0]
y = np.log1p(df_train["price"])
merge: pd.DataFrame = pd.concat([df_train, df_test])

del df_train
del df_test
gc.collect()

handle_missing_inplace(merge)
print('[{}] Finished to handle missing'.format(time.time() - start_time))

cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))

cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])
print('[{}] Finished count vectorize category_name`'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                     ngram_range=(1, 3),
                     stop_words='english')
X_description = tv.fit_transform(merge['item_description'])
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]
model = Ridge(solver="sag", fit_intercept=True, alpha=3.5, random_state=666)
model.fit(X, y)
print('[{}] Finished to train ridge'.format(time.time() - start_time))
predsR = model.predict(X=X_test)
print('[{}] Finished to predict ridge'.format(time.time() - start_time))
predsR = np.expm1(predsR)
predsR = predsR * 0.14
submission["price"] += predsR

model = Ridge(solver="sag", fit_intercept=False, alpha=1.5, random_state=666)
model.fit(X, y)
print('[{}] Finished to train ridge'.format(time.time() - start_time))
predsRR = model.predict(X=X_test)
print('[{}] Finished to predict ridge'.format(time.time() - start_time))
predsRR = np.expm1(predsRR)
predsRR = predsRR * 0.06
submission["price"] += predsRR

submission.to_csv("submission.csv", index=False)