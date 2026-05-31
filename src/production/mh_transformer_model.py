from __future__ import annotations

from dataclasses import dataclass


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "TensorFlow is required for training/inference.\n"
            "Install one of:\n"
            "  - tensorflow-cpu (edge CPU)\n"
            "  - tensorflow (GPU)\n"
        ) from e
    import tensorflow as tf

    return tf


@dataclass(frozen=True)
class MHTransformerHP:
    k: int = 10
    t: int = 25
    f: int = 7
    h: int = 5
    d_model: int = 96
    n_heads: int = 4
    ff_dim: int = 192
    n_layers: int = 3
    dropout: float = 0.15
    lr: float = 3e-4


def temporal_focal_loss(*, gamma: float = 2.0, alpha: float = 0.25):
    tf = _require_tensorflow()

    def _loss(y_true, y_pred):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        ce = -y_true * tf.math.log(y_pred)  # (B, H, K)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)  # (B, H, 1)
        w = alpha * tf.pow(1.0 - p_t, gamma)  # (B, H, 1)
        step_loss = tf.reduce_sum(w * ce, axis=-1)  # (B, H)
        return tf.reduce_mean(step_loss)

    _loss.__name__ = f"temporal_focal_g{gamma}_a{alpha}"
    return _loss


def build_mh_transformer(hp: MHTransformerHP):
    tf = _require_tensorflow()
    keras = tf.keras
    layers = keras.layers

    class MaskedGlobalAveragePooling(layers.Layer):
        def call(self, x, mask):
            summed = tf.reduce_sum(x * mask, axis=1)
            count = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-8)
            return summed / count

    def encoder_block(x):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        att = layers.MultiHeadAttention(
            num_heads=hp.n_heads,
            key_dim=hp.d_model // hp.n_heads,
            dropout=hp.dropout,
        )(x1, x1)
        att = layers.Dropout(hp.dropout)(att)
        x2 = layers.Add()([x, att])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ff = layers.Dense(hp.ff_dim, activation="relu")(x3)
        ff = layers.Dropout(hp.dropout)(ff)
        ff = layers.Dense(hp.d_model)(ff)
        return layers.Add()([x2, ff])

    inp_cells = keras.Input((hp.k, hp.t, hp.f), name="cells", dtype="float32")
    inp_mask = keras.Input((hp.k,), name="mask", dtype="float32")

    # Per-cell temporal encoder (shared): reshape to (B*K, T, F)
    x = tf.reshape(inp_cells, (-1, hp.t, hp.f))
    x = layers.Dense(hp.d_model)(x)

    pos = tf.range(start=0, limit=hp.t, delta=1)
    pos_emb = layers.Embedding(input_dim=hp.t, output_dim=hp.d_model)(pos)  # (T, D)
    x = x + pos_emb

    for _ in range(hp.n_layers):
        x = encoder_block(x)

    x = layers.GlobalAveragePooling1D()(x)  # (B*K, D)
    phi = tf.reshape(x, (-1, hp.k, hp.d_model))  # (B, K, D)

    # DeepSet context pooling over cells
    mask_exp = layers.Reshape((hp.k, 1))(inp_mask)
    z = MaskedGlobalAveragePooling()(phi, mask_exp)  # (B, D)

    # Horizon embedding
    h_ids = tf.range(hp.h)
    h_emb = layers.Embedding(input_dim=hp.h, output_dim=hp.d_model)(h_ids)  # (H, D)

    # Broadcast tensors to (B, K, H, D)
    phi4 = tf.tile(tf.expand_dims(phi, axis=2), [1, 1, hp.h, 1])
    z4 = tf.tile(tf.reshape(z, (-1, 1, 1, hp.d_model)), [1, hp.k, hp.h, 1])
    he4 = tf.tile(tf.reshape(h_emb, (1, 1, hp.h, hp.d_model)), [tf.shape(phi)[0], hp.k, 1, 1])

    u = layers.Concatenate(axis=-1)([phi4, z4, he4])  # (B, K, H, 3D)
    u = layers.Dense(hp.d_model, activation="relu")(u)
    u = layers.Dropout(hp.dropout)(u)
    logits = layers.Dense(1)(u)  # (B, K, H, 1)
    logits = tf.transpose(tf.squeeze(logits, axis=-1), perm=[0, 2, 1])  # (B, H, K)

    # Masked softmax per horizon
    mask_h = tf.expand_dims(inp_mask, axis=1)  # (B, 1, K)
    logits = logits + (1.0 - mask_h) * (-1e9)
    probs = layers.Softmax(dtype="float32", name="cell_probs_h")(logits)

    model = keras.Model([inp_cells, inp_mask], probs, name="MH_Transformer_DeepSet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.lr),
        loss=temporal_focal_loss(),
        metrics=[keras.metrics.CategoricalAccuracy(name="top1_acc")],
    )
    return model

