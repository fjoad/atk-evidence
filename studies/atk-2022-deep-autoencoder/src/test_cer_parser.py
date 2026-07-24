from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from cer_parser import (
    daily_profiles,
    decode_day_time_code,
    iter_authorized_cer_zip_chunks,
    read_allocation_table,
    read_cer_text,
    residential_meter_ids,
)


def meter_day_text(meter_id: int, day: int, slots: range) -> str:
    return "".join(
        f"{meter_id} {day * 100 + slot} {slot / 10:.1f}\n" for slot in slots
    )


class CerParserTests(unittest.TestCase):
    def test_manifest_day_time_mapping_and_dst_extra_slots(self) -> None:
        decoded = decode_day_time_code(pd.Series([101, 148, 149, 150, 201]))
        self.assertEqual(str(decoded.loc[0, "timestamp"]), "2009-01-01 00:00:00")
        self.assertEqual(str(decoded.loc[1, "timestamp"]), "2009-01-01 23:30:00")
        self.assertEqual(str(decoded.loc[2, "timestamp"]), "2009-01-02 00:00:00")
        self.assertEqual(str(decoded.loc[4, "timestamp"]), "2009-01-02 00:00:00")
        self.assertEqual(decoded["is_dst_extra_slot"].tolist(), [False, False, True, True, False])

    def test_text_read_validates_values(self) -> None:
        text = "1001 101 0.25\n1001 102 0.50\n1002 101 1.25\n"
        frame = read_cer_text(io.StringIO(text))
        self.assertEqual(frame.shape, (3, 8))
        self.assertEqual(frame.loc[0, "meter_id"], 1001)
        self.assertEqual(frame.loc[1, "half_hour_slot"], 2)
        with self.assertRaises(ValueError):
            read_cer_text(io.StringIO("1001 101 -0.1\n"))

    def test_complete_profiles_exclude_missing_and_50_slot_days(self) -> None:
        text = (
            meter_day_text(1001, 1, range(1, 49))
            + meter_day_text(1001, 2, range(1, 47))
            + meter_day_text(1002, 3, range(1, 51))
        )
        profiles = daily_profiles(read_cer_text(io.StringIO(text)))
        self.assertEqual(profiles.shape, (1, 50))
        self.assertEqual(profiles.loc[0, "meter_id"], 1001)
        self.assertEqual(profiles.loc[0, "hh_01"], 0.1)
        self.assertEqual(profiles.loc[0, "hh_48"], 4.8)
        self.assertEqual(
            profiles.attrs["cer_profile_stats"],
            {
                "meter_days_total": 3,
                "complete_48_slot_days": 1,
                "excluded_days": 2,
                "dst_extra_slot_days": 1,
            },
        )

    def test_duplicate_slot_rejected(self) -> None:
        frame = read_cer_text(io.StringIO("1001 101 0.1\n1001 101 0.2\n"))
        with self.assertRaises(ValueError):
            daily_profiles(frame)

    def test_all_iset_day_interpretations_are_explicit(self) -> None:
        base = read_cer_text(
            io.StringIO(
                meter_day_text(1001, 1, range(1, 49))
                + meter_day_text(1001, 2, range(1, 47))
                + meter_day_text(1002, 3, range(1, 51))
            )
        )
        trimmed = daily_profiles(base, policy="trim_extra")
        self.assertEqual(
            trimmed[["meter_id", "day_number"]].values.tolist(),
            [[1001, 1], [1002, 3]],
        )
        interpolated = daily_profiles(base, policy="interpolate_grid")
        self.assertEqual(interpolated.shape[0], 3)
        short_day = interpolated.loc[
            (interpolated["meter_id"] == 1001)
            & (interpolated["day_number"] == 2)
        ].iloc[0]
        self.assertAlmostEqual(short_day["hh_47"], 4.6)
        self.assertAlmostEqual(short_day["hh_48"], 4.6)

        duplicated = pd.concat(
            [
                read_cer_text(
                    io.StringIO(meter_day_text(1003, 4, range(1, 49)))
                ),
                read_cer_text(io.StringIO("1003 405 9.5\n")),
            ],
            ignore_index=True,
        )
        aggregated = daily_profiles(
            duplicated,
            policy="aggregate_duplicate_slots",
        )
        self.assertEqual(aggregated.shape[0], 1)
        self.assertAlmostEqual(aggregated.loc[0, "hh_05"], 5.0)

    def test_allocation_table_and_residential_selection(self) -> None:
        allocation_text = (
            "\ufeffMeter ID\tCode\tResidential stimulus\n"
            "1003\t2\t\n"
            "1001\t1\tA\n"
            "1002\t1\tB\n"
            "1002\t1\tB\n"
        )
        allocation = read_allocation_table(io.StringIO(allocation_text))
        self.assertEqual(allocation.to_dict("list"), {
            "meter_id": [1001, 1002, 1003],
            "allocation_code": [1, 1, 2],
        })
        self.assertEqual(residential_meter_ids(allocation).tolist(), [1001, 1002])

    def test_conflicting_allocation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            read_allocation_table(io.StringIO("ID\tCode\n1001\t1\n1001\t2\n"))

    def test_chunked_zip_ingestion_filters_residential_ids(self) -> None:
        payload = (
            "1001 101 0.1\n"
            "2001 101 0.2\n"
            "1001 102 0.3\n"
            # Slot 95 is malformed but belongs to an unused meter.  The
            # paper-declared residential selection occurs before time-code
            # validation, so it must not block meter 1001.
            "2001 195 0.4\n"
            "1001 149 0.5\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "File1.txt.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("File1.txt", payload)
            chunks = list(
                iter_authorized_cer_zip_chunks(
                    archive_path,
                    residential_ids=[1001],
                    chunksize=2,
                )
            )
        combined = pd.concat(chunks, ignore_index=True)
        self.assertEqual(combined["meter_id"].unique().tolist(), [1001])
        self.assertEqual(combined["half_hour_slot"].tolist(), [1, 2, 49])
        self.assertEqual(chunks[0].attrs["source_member"], "File1.txt")
        self.assertEqual(
            [chunk.attrs["source_chunk"] for chunk in chunks], [0, 1, 2]
        )

    def test_chunked_zip_ingestion_rejects_invalid_selected_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "File1.txt.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("File1.txt", "1001 195 0.1\n")
            with self.assertRaisesRegex(ValueError, "slot must be between 1 and 50"):
                list(
                    iter_authorized_cer_zip_chunks(
                        archive_path,
                        residential_ids=[1001],
                        chunksize=2,
                    )
                )

    def test_invalid_slot_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_day_time_code(pd.Series([151]))
        with self.assertRaises(ValueError):
            decode_day_time_code(pd.Series([101.5]))


if __name__ == "__main__":
    unittest.main()
