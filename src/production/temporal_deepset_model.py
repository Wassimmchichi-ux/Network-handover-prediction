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
class TemporalDeepSetHP:
    max_cells: int = 10
    obs_steps: int = 25
    n_feats: int = 3
    loss_type: str = "focal"  # "focal" | "cce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    label_smooth: float = 0.1
    lstm_units: int = 64
    phi_dim: int = 64
    phi_layers: int = 2
    dropout: float = 0.25
    lr_init: float = 1e-3


def focal_loss(*, gamma: float = 2.0, alpha: float = 0.25):
    tf = _require_tensorflow()

    def _loss(y_true, y_pred):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        ce = -y_true * tf.math.log(y_pred)  # (B, C)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)  # (B, 1)
        fw = alpha * tf.pow(1.0 - p_t, gamma)  # (B, 1)
        return tf.reduce_mean(tf.reduce_sum(fw * ce, axis=-1))

    _loss.__name__ = f"focal_g{gamma}_a{alpha}"
    return _loss


def build_temporal_deepset(hp: TemporalDeepSetHP):
    tf = _require_tensorflow()
    keras = tf.keras
    layers = keras.layers

    class MaskedGlobalAveragePooling(layers.Layer):
        def call(self, phi, mask):
            summed = tf.reduce_sum(phi * mask, axis=1)
            count = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-8)
            return summed / count

    C, W, F, D = hp.max_cells, hp.obs_steps, hp.n_feats, hp.phi_dim

    inp_cells = keras.Input((C, W, F), name="cells", dtype="float32")
    inp_mask = keras.Input((C,), name="mask", dtype="float32")

    trend = layers.TimeDistributed(
        layers.LSTM(hp.lstm_units, return_sequences=False),
        name="td_lstm",
    )(inp_cells)

    phi = trend
    for i in range(hp.phi_layers):
        phi = layers.TimeDistributed(
            layers.Dense(D, activation="relu"),
            name=f"phi_{i}",
        )(phi)
        phi = layers.TimeDistributed(
            layers.Dropout(hp.dropout),
            name=f"phi_drop_{i}",
        )(phi)

    mask_exp = layers.Reshape((C, 1), name="mask_exp")(inp_mask)
    z = MaskedGlobalAveragePooling(name="pool")(phi, mask_exp)
    z_tiled = layers.RepeatVector(C, name="z_tile")(z)

    rho = layers.Concatenate(axis=-1, name="rho_cat")([phi, z_tiled])
    rho = layers.TimeDistributed(
        layers.Dense(D, activation="relu"),
        name="rho",
    )(rho)
    rho = layers.TimeDistributed(layers.Dropout(hp.dropout), name="rho_drop")(rho)

    logits = layers.TimeDistributed(layers.Dense(1, use_bias=True), name="scorer")(rho)
    logits = layers.Reshape((C,), name="logits")(logits)
    probs = layers.Softmax(dtype="float32", name="cell_probs")(
        layers.Add(name="pad_mask")([logits, (1.0 - inp_mask) * (-1e9)])
    )

    model = keras.Model([inp_cells, inp_mask], probs, name="TemporalDeepSet")

    if hp.loss_type == "focal":
        loss = focal_loss(gamma=hp.focal_gamma, alpha=hp.focal_alpha)
    elif hp.loss_type == "cce":
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=hp.label_smooth)
    else:
        raise ValueError(f"Unknown loss_type: {hp.loss_type}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.lr_init),
        loss=loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="top1_acc"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


def load_temporal_deepset(model_path: str):
    tf = _require_tensorflow()
    keras = tf.keras
    layers = keras.layers

    class MaskedGlobalAveragePooling(layers.Layer):
        def call(self, phi, mask):
            summed = tf.reduce_sum(phi * mask, axis=1)
            count = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-8)
            return summed / count

    return keras.models.load_model(
        model_path,
        custom_objects={"MaskedGlobalAveragePooling": MaskedGlobalAveragePooling},
        compile=False,
    )

