import numpy as np


def softmax(vector):
    """
    vector: np.array of shape (n, m)

    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    """
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_


def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # Step 1: Compute attention scores using multiplicative formula
    attention_scores = np.dot(
        np.dot(decoder_hidden_state.T, W_mult), encoder_hidden_states
    )

    # Step 2: Apply softmax to get weights
    attention_weights = softmax(attention_scores)

    # Step 3: Compute weighted sum of encoder states
    attention_vector = np.dot(attention_weights, encoder_hidden_states.T).T

    return attention_vector


def additive_attention(
    decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec
):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # Step 1: Transform all encoder states at once
    enc_part = np.dot(W_add_enc, encoder_hidden_states)  # (n_features_int, n_states)

    # Step 2: Transform decoder state
    dec_part = np.dot(W_add_dec, decoder_hidden_state)  # (n_features_int, 1)

    # Step 3: Add decoder part to each encoder column (broadcasting)
    added = enc_part + dec_part  # (n_features_int, n_states)

    # Step 4: Apply tanh activation
    activated = np.tanh(added)  # (n_features_int, n_states)

    # Step 5: Compute attention scores for all states
    attention_scores = np.dot(v_add.T, activated)  # (1, n_states)

    # Step 6: Apply softmax to get weights
    attention_weights = softmax(attention_scores)

    # Step 7: Compute weighted sum of encoder states
    attention_vector = np.dot(attention_weights, encoder_hidden_states.T).T

    return attention_vector
