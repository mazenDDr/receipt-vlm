from receipt_vlm.eval import fields, normalize
from receipt_vlm.eval.parse import extract_json


def test_flatten_handles_single_item_list_items_and_nested_subs():
    single = {"menu": {"nm": "EGG TART", "price": "13,000"}}
    many = {
        "menu": [
            {"nm": "LATTE", "sub": [{"nm": "Less Ice"}, {"nm": "70%"}]},
            {"nm": "TOAST"},
        ],
        "total": {"cashprice": ["50,000", "10,000"]},
    }
    assert fields.flatten(single) == [("menu.nm", "EGG TART"), ("menu.price", "13,000")]
    assert fields.flatten(many) == [
        ("menu.nm", "LATTE"),
        ("menu.sub.nm", "Less Ice"),
        ("menu.sub.nm", "70%"),
        ("menu.nm", "TOAST"),
        ("total.cashprice", "50,000"),
        ("total.cashprice", "10,000"),
    ]


def test_kind_uses_the_cord_table_and_falls_back_on_the_key_name():
    assert fields.kind("menu.nm") == "text"
    assert fields.kind("menu.sub.nm") == "text"
    assert fields.kind("menu.price") == "numeric"
    assert fields.kind("menu.num") == "numeric"
    assert fields.kind("total.menuqty_cnt") == "numeric"
    assert fields.kind("total.tip_price") == "numeric"  # not a CORD key
    assert fields.kind("store.address") == "text"  # not a CORD key


def test_strict_only_strips():
    assert normalize.strict("  16,500 ") == "16,500"
    assert normalize.strict("16.500") != normalize.strict("16,500")


def test_lenient_forgives_separators_and_currency_but_not_digits():
    assert (
        normalize.lenient("Rp 16.500")
        == normalize.lenient("@16,500")
        == normalize.lenient("16500")
        == "16500"
    )
    assert normalize.lenient("70000.00") == "70000"
    assert normalize.lenient("-9,545") == "-9545"
    assert normalize.lenient("16,500") != normalize.lenient("16,800")
    assert normalize.lenient("Less  Ice") == "less ice"


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"total": {"total_price": "60.000"}}\n```') == {
        "total": {"total_price": "60.000"}
    }
    assert extract_json('Here it is: {"menu": {"nm": "A"}} Done.') == {"menu": {"nm": "A"}}


def test_extract_json_does_not_repair_broken_output():
    assert extract_json('{"menu": [{"nm": "A"}, {"nm": ') is None
    assert extract_json("no json here") is None
    assert extract_json("[1, 2]") is None


def test_extract_json_merges_repeated_keys_into_a_list():
    raw = '{"menu": {"nm": "A"}, "menu": {"nm": "B"}, "total": {"total_price": "5"}}'
    assert extract_json(raw) == {"menu": [{"nm": "A"}, {"nm": "B"}], "total": {"total_price": "5"}}
