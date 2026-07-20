from __future__ import annotations

import io
import unittest

import pandas as pd

from cer_parser import daily_profiles, decode_day_time_code, read_cer_text


class CerParserTests(unittest.TestCase):
    def test_manifest_day_time_mapping(self) -> None:
        decoded = decode_day_time_code(pd.Series([101, 148, 201]))
        self.assertEqual(str(decoded.loc[0, "timestamp"]), "2009-01-01 00:00:00")
        self.assertEqual(str(decoded.loc[1, "timestamp"]), "2009-01-01 23:30:00")
        self.assertEqual(str(decoded.loc[2, "timestamp"]), "2009-01-02 00:00:00")

    def test_text_read_and_pivot(self) -> None:
        text = "1001 101 0.25\n1001 102 0.50\n1002 101 1.25\n"
        frame = read_cer_text(io.StringIO(text))
        profiles = daily_profiles(frame)
        self.assertEqual(profiles.shape, (2, 50))
        self.assertEqual(profiles.loc[0, "hh_01"], 0.25)
        self.assertEqual(profiles.loc[0, "hh_02"], 0.50)
        self.assertTrue(pd.isna(profiles.loc[0, "hh_48"]))

    def test_invalid_slot_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_day_time_code(pd.Series([149]))


if __name__ == "__main__":
    unittest.main()

