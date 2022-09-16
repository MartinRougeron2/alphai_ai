import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate, Layer
from collections import defaultdict
from collections import namedtuple, OrderedDict
from itertools import chain


from keras.layers import Input, Lambda
from keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.ops.lookup_ops import TextFileInitializer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Reshape, Flatten
from tensorflow.python.keras.layers import Embedding, Lambda

from tensorflow.python.ops.lookup_ops import StaticHashTable
from ..layers import (
    Hash,
    SparseFeat,
    VarLenSparseFeat,
    DenseFeat,
    WeightedSequenceLayer,
    SequencePoolingLayer,
    NoMask,
)


def get_inputs_list(inputs):
    return list(
        chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs))))
    )


def create_embedding_dict(
    sparse_feature_columns,
    varlen_sparse_feature_columns,
    seed,
    l2_reg,
    prefix="sparse_",
    seq_mask_zero=True,
):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(
            feat.vocabulary_size,
            feat.embedding_dim,
            embeddings_initializer=feat.embeddings_initializer,
            embeddings_regularizer=l2(l2_reg),
            name=prefix + "_emb_" + feat.embedding_name,
        )
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(
                feat.vocabulary_size,
                feat.embedding_dim,
                embeddings_initializer=feat.embeddings_initializer,
                embeddings_regularizer=l2(l2_reg),
                name=prefix + "_seq_emb_" + feat.name,
                mask_zero=seq_mask_zero,
            )
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def get_embedding_vec_list(
    embedding_dict,
    input_dict,
    sparse_feature_columns,
    return_feat_list=(),
    mask_feat_list=(),
):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(
                    fg.vocabulary_size,
                    mask_zero=(feat_name in mask_feat_list),
                    vocabulary_path=fg.vocabulary_path,
                )(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list


def create_embedding_matrix(
    feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True
):

    sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
        if feature_columns
        else []
    )
    varlen_sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
        if feature_columns
        else []
    )
    sparse_emb_dict = create_embedding_dict(
        sparse_feature_columns,
        varlen_sparse_feature_columns,
        seed,
        l2_reg,
        prefix=prefix + "sparse",
        seq_mask_zero=seq_mask_zero,
    )
    return sparse_emb_dict


def embedding_lookup(
    sparse_embedding_dict,
    sparse_input_dict,
    sparse_feature_columns,
    return_feat_list=(),
    mask_feat_list=(),
    to_list=False,
):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            if fc.use_hash:
                lookup_idx = Hash(
                    fc.vocabulary_size,
                    mask_zero=(feature_name in mask_feat_list),
                    vocabulary_path=fc.vocabulary_path,
                )(sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(
                sparse_embedding_dict[embedding_name](lookup_idx)
            )
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(
    embedding_dict, sequence_input_dict, varlen_sparse_feature_columns
):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(
                fc.vocabulary_size, mask_zero=True, vocabulary_path=fc.vocabulary_path
            )(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](
            lookup_idx
        )
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(
    embedding_dict, features, varlen_sparse_feature_columns, to_list=False
):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [
                        embedding_dict[feature_name],
                        features[feature_length_name],
                        features[fc.weight_name],
                    ]
                )
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]]
            )
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(
                    weight_normalization=fc.weight_norm, supports_masking=True
                )([embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    dense_feature_columns = (
        list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
        if feature_columns
        else []
    )
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list


def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c


def input_from_feature_columns(
    features,
    feature_columns,
    l2_reg,
    seed,
    prefix="",
    seq_mask_zero=True,
    support_dense=True,
    support_group=False,
):
    sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
        if feature_columns
        else []
    )
    varlen_sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
        if feature_columns
        else []
    )

    embedding_matrix_dict = create_embedding_matrix(
        feature_columns, l2_reg, seed, prefix=prefix, seq_mask_zero=seq_mask_zero
    )
    group_sparse_embedding_dict = embedding_lookup(
        embedding_matrix_dict, features, sparse_feature_columns
    )
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(
        embedding_matrix_dict, features, varlen_sparse_feature_columns
    )
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(
        sequence_embed_dict, features, varlen_sparse_feature_columns
    )
    group_embedding_dict = mergeDict(
        group_sparse_embedding_dict, group_varlen_sparse_embedding_dict
    )
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


def build_input_features(feature_columns, prefix=""):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype
            )
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype
            )
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(
                shape=(fc.maxlen,), name=prefix + fc.name, dtype=fc.dtype
            )
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(
                    shape=(fc.maxlen, 1), name=prefix + fc.weight_name, dtype="float32"
                )
            if fc.length_name is not None:
                input_features[fc.length_name] = Input(
                    (1,), name=prefix + fc.length_name, dtype="int32"
                )

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features
