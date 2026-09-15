from receipt_vlm.quant import awq


def test_calibration_rows_carry_real_input_ids_and_the_receipt_index():
    seen = []

    def input_ids_for(i):
        seen.append(i)
        return [100 + i] * (i + 1)

    rows = awq.calibration_rows(3, input_ids_for)
    # llm-compressor only skips its own re-tokenizing when an input_ids column is present
    assert rows == {"i": [0, 1, 2], "input_ids": [[100], [101, 101], [102, 102, 102]]}
    assert seen == [0, 1, 2]
