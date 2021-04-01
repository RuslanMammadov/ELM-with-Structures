from src.layers.structured_layers import LowRank
from src.elm.base_elm import ELM


def main():
    test_elm_with_low_rank_layer()


def test_elm_with_low_rank_layer():
    inputs = [[1, 2, 3], [3, 4, 5]]
    elm = ELM()
    elm.add_dense_layer(10, input_dim=3)
    elm.add_layer(LowRank(35, 5, use_bias=True))
    elm.add_dense_layer(20)
    elm.add_layer(LowRank(3, 4, use_bias=False))
    elm.compile()
    features = elm.get_transformed_features(inputs)
    assert features.shape[0] == 2
    assert features.shape[1] == 3

    print("Low Rank test was succesful!")


if __name__ == '__main__':
    main()
