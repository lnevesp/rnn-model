
import time

start_time = time.time()
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


class PreProcessing():
    NUM_BRANDS = 4000
    NUM_CATEGORIES = 1000
    NAME_MIN_DF = 10
    MAX_FEATURES_ITEM_DESCRIPTION = 40000


    def handle_missing_inplace(self, dataset):
        dataset['category_name'].fillna(value='missing', inplace=True)
        dataset['brand_name'].fillna(value='missing', inplace=True)
        dataset['item_description'].fillna(value='missing', inplace=True)
        dataset.item_description.replace('No description yet', "missing", inplace=True)

    def cutting(self, dataset):
        pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
        dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
        pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
        dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'


    def to_categorical(self, dataset):
        dataset['category_name'] = dataset['category_name'].astype('category')
        dataset['brand_name'] = dataset['brand_name'].astype('category')
        dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


    df_train = train = pd.read_csv('../input/train.tsv', sep='\t')
    df_test = test = pd.read_csv('../input/test.tsv', sep='\t')

    train['target'] = np.log1p(train['price'])

    print(train.shape)
    print('5 folds scaling the test_df')
    print(test.shape)
    test_len = test.shape[0]


    def simulate_test(self, test):
        if test.shape[0] < 800000:
            indices = np.random.choice(test.index.values, 2800000)
            test_ = pd.concat([test, test.iloc[indices]], axis=0)
            return test_.copy()
        else:
            return test


    test = simulate_test(self, test)
    print('new shape ', test.shape)
    print('[{}] Finished scaling test set...'.format(time.time() - start_time))

    # HANDLE MISSING VALUES
    print("Handling missing values...")


    def handle_missing(self, dataset):
        dataset.category_name.fillna(value="missing", inplace=True)
        dataset.brand_name.fillna(value="missing", inplace=True)
        dataset.item_description.fillna(value="missing", inplace=True)
        return (dataset)


    train = handle_missing(train)
    test = handle_missing(test)
    print(train.shape)
    print(test.shape)

    print('[{}] Finished handling missing data...'.format(time.time() - start_time))

    # PROCESS CATEGORICAL DATA
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

    print("Handling categorical variables...")
    le = LabelEncoder()

    le.fit(np.hstack([train.category_name, test.category_name]))
    train['category'] = le.transform(train.category_name)
    test['category'] = le.transform(test.category_name)

    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train['brand'] = le.transform(train.brand_name)
    test['brand'] = le.transform(test.brand_name)
    del le, train['brand_name'], test['brand_name']

    print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
    train.head(3)

    # PROCESS TEXT: RAW
    print("Text to seq process...")
    print("   Fitting tokenizer...")

    from keras.preprocessing.text import Tokenizer

    raw_text = np.hstack([train.category_name.str.lower(),
                          train.item_description.str.lower(),
                          train.name.str.lower()])

    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    print("   Transforming text to seq...")
    train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
    test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
    train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
    test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
    train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
    train.head(3)

    print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))

    # EXTRACT DEVELOPTMENT TEST
    from sklearn.model_selection import train_test_split

    dtrain, dvalid = train_test_split(train, random_state=666, train_size=0.99)
    print(dtrain.shape)
    print(dvalid.shape)

    # EMBEDDINGS MAX VALUE
    # Base on the histograms, we select the next lengths
    MAX_NAME_SEQ = 20  # 17
    MAX_ITEM_DESC_SEQ = 60  # 269
    MAX_CATEGORY_NAME_SEQ = 20  # 8
    MAX_TEXT = np.max([np.max(train.seq_name.max())
                          , np.max(test.seq_name.max())
                          , np.max(train.seq_category_name.max())
                          , np.max(test.seq_category_name.max())
                          , np.max(train.seq_item_description.max())
                          , np.max(test.seq_item_description.max())]) + 2
    MAX_CATEGORY = np.max([train.category.max(), test.category.max()]) + 1
    MAX_BRAND = np.max([train.brand.max(), test.brand.max()]) + 1
    MAX_CONDITION = np.max([train.item_condition_id.max(),
                            test.item_condition_id.max()]) + 1

    print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))

    # KERAS DATA DEFINITION



    def get_keras_data(dataset):
        X = {
            'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
            , 'item_desc': pad_sequences(dataset.seq_item_description
                                         , maxlen=MAX_ITEM_DESC_SEQ)
            , 'brand': np.array(dataset.brand)
            , 'category': np.array(dataset.category)
            , 'category_name': pad_sequences(dataset.seq_category_name
                                             , maxlen=MAX_CATEGORY_NAME_SEQ)
            , 'item_condition': np.array(dataset.item_condition_id)
            , 'num_vars': np.array(dataset[["shipping"]])
        }
        return X


    X_train = get_keras_data(dtrain)
    X_valid = get_keras_data(dvalid)
    X_test = get_keras_data(test)

    print('[{}] Finished DATA PREPARARTION...'.format(time.time() - start_time))

