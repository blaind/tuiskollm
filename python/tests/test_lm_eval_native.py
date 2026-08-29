import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "lm_eval_tuisko_native.py"
SPEC = importlib.util.spec_from_file_location("lm_eval_tuisko_native", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


class PrefixTokenizer(Tokenizer):
    def encode(self, text, add_special_tokens=False):
        del text
        assert not add_special_tokens
        return [99, 120]


def test_encode_pair_matches_lm_eval_boundary_and_moves_trailing_spaces():
    assert MODULE.encode_pair(Tokenizer(), "", "x") == ([99], [ord("x")])
    assert MODULE.encode_pair(Tokenizer(), "a  ", "b") == (
        [ord("a")],
        [ord(" "), ord(" "), ord("b")],
    )
    assert MODULE.encode_pair(PrefixTokenizer(), "", "ignored") == ([99], [120])


def test_batch_size_and_left_context_follow_native_route_limits():
    assert MODULE.parse_batch_size("auto") == 8
    assert MODULE.parse_batch_size("16") == 8
    assert MODULE.parse_batch_size(4) == 4
    assert MODULE.fit_context([1, 2, 3, 4], [5, 6], 4) == ([3, 4], [5, 6])


def test_adjacent_groups_preserve_order_and_cap_branches_at_eight():
    encoded = [([1], [index]) for index in range(9)] + [([2], [10]), ([1], [11])]
    groups = list(MODULE.adjacent_context_groups(encoded))

    assert [len(group) for _, group in groups] == [8, 1, 1, 1]
    assert [context for context, _ in groups] == [[1], [1], [2], [1]]
    assert [index for _, group in groups for index, _ in group] == list(range(11))

    groups = list(MODULE.adjacent_context_groups(encoded[:5], maximum=4))
    assert [len(group) for _, group in groups] == [4, 1]
